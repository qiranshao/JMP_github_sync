# version for github 20230409 # add kmeans
cd("/Users/shaoqiran/Documents/Documents - MacBook Pro/Rice University/JMP/Latest benchmark files/50_games_data_for_julia_new")
#using Distributed
#addprocs(6)
#using Pkg; Pkg.add("LightGraphs")
#using LightGraphs
#using Graphs, SimpleWeightedGraphs
#Pkg.add("GraphPlot")
#using GraphPlot
######
#using BlackBoxOptim
using Distributions # for generating fake data from certain distributions
using DataFrames # store data in dataframe
using DataFramesMeta # allow more flexible command for dataframe
using LinearAlgebra # for "\" operator (least squCHe regression)
using Random # set seed
using DelimitedFiles # for exporting txt files
using Dates ## output with dates
using Statistics # calculate mean, var etc.
using Plots # plots for conditional expectations etc.
using Missings ## allow code to be robust to missing values
#Pkg.add("DecisionTree")
#using DecisionTree ## random forest regression for conditional expectation
using CSV # for export csv files
using StatsBase # support package for Statistics
using SharedArrays
using Dates

## 1 prepare dataframes
Country_List = ["DE", "GB", "FR", "BR", "CA", "AU", "SE", "DK", "NL", "NO"]

## number of players who played top 50 games in 10 countries
#(53113 + 59831 + 27387 + 14819 + 37926 + 30144 + 18191 + 11129 + 12237 + 10730) / (61754 + 73543 + 32932 + 17900 + 47030 + 36588 + 21633 + 13065 + 14492 + 12483)

## average number of total recently released games in 10 countries
#(318 + 320 + 299 + 289 + 312 + 309 + 284 + 265 + 277 + 248)/10

# index countries
const num_cc = length(Country_List)
const num_games = 50
#@time CSV.read("AU_top_game_info.csv", DataFrame)
#@time CSV.read("DE_Friends_Close.csv", DataFrame)
cc=1
#  dataframes prepared to be used in the following functions, with indeces for country and genre, all countries
function Key_DataFrames_generator(num_cc)
    friends = DataFrame(country_index = Any[], steamid_a = Any[], steamid_b = Any[])
    plays = DataFrame(country_index = Any[], steamid = Any[], appid = Any[], playtime_2weeks = Any[])
    groups = DataFrame(country_index = Any[], steamid = Any[], groupid = Any[])
    total_game_time = DataFrame(country_index = Any[], steamid = Any[], total_game_time = Any[])
    games_played = DataFrame(country_index = Any[], steamid = Any[], appid = Any[])
    dfmt = dateformat"yyyy-mm-dd HH:MM:SS"
    top_game_info = DataFrame(country_index = Any[], appid = Any[], Title = Any[], Price = Any[], Rating = Any[], Required_Age = Any[], Is_Multiplayer = Any[], Release_Date = Any[], New_Rating = Any[], D_P_Match = Any[], Developer_Employee = Any[], Min_Ram = Any[], Max_Ram = Any[], Min_DirectX = Any[], Max_DirectX = Any[], Min_Space = Any[], Max_Space = Any[])
    date_obs ="2014-08-14 00:00:00"
    date_obs = Date(date_obs,dfmt)
    for cc in 1:num_cc
        println("cc = $(cc)")
        # friends tables
        friends_temp = CSV.read("$(Country_List[cc])_Friends_Close.csv", DataFrame)
        friends_n_rows=nrow(friends_temp)
        f_cc_index = DataFrame(country_index = Array{Int64}(cc * ones(friends_n_rows)))
        friends_with_cc_index = hcat(f_cc_index, friends_temp)
        friends = vcat(friends, friends_with_cc_index)
        # plays tables
        plays_temp = CSV.read("$(Country_List[cc])_temp_unique_close.csv", DataFrame)
        plays_n_rows=nrow(plays_temp)
        p_cc_index = DataFrame(country_index = Array{Int64}(cc * ones(plays_n_rows)))
        plays_with_cc_index = hcat(p_cc_index, plays_temp)
        plays = vcat(plays, plays_with_cc_index)
        # groups tables
        groups_temp = CSV.read("$(Country_List[cc])_groups.csv", DataFrame)
        groups_n_rows=nrow(groups_temp)
        g_cc_index = DataFrame(country_index = Array{Int64}(cc * ones(groups_n_rows)))
        groups_with_cc_index = hcat(g_cc_index, groups_temp)
        groups = vcat(groups, groups_with_cc_index)
        # total_game_time tables
        total_game_time_temp = CSV.read("$(Country_List[cc])_total_game_time.csv", DataFrame)
        total_game_time_n_rows=nrow(total_game_time_temp)
        tgt_cc_index = DataFrame(country_index = Array{Int64}(cc * ones(total_game_time_n_rows)))
        total_game_time_with_cc_index = hcat(tgt_cc_index, total_game_time_temp)
        total_game_time = vcat(total_game_time, total_game_time_with_cc_index)
        # top_game_info tables
        top_game_info_temp = CSV.read("$(Country_List[cc])_top_game_info.csv", DataFrame)
        top_game_info_n_rows=nrow(top_game_info_temp)
        tgi_cc_index = DataFrame(country_index = Array{Int64}(cc * ones(top_game_info_n_rows)))
        top_game_info_with_cc_index = hcat(tgi_cc_index, top_game_info_temp)
        top_game_info = vcat(top_game_info, top_game_info_with_cc_index)
    end
    num_games = 50
    return top_game_info, num_games, friends, plays, groups, total_game_time, top_game_info, date_obs
end
#@time (top_game_info, num_games, friends, plays, groups, total_game_time, top_game_info, date_obs) = Key_DataFrames_generator(num_cc)
#print(top_game_info)
#print(names(top_game_info))
## 2 gen datasets A_cc... for model for country cc
## 2.1 gen info about play record, friend matrix, num of players, game info for cc
#cc = 4 # try this country
function gen_cc(cc,top_game_info,num_games,friends,plays)
    # set up the environment under cc
    @linq cc_plays = plays |>
    where(:country_index .== cc)
    cc_num_players = nrow(cc_plays) # one game per player, so num_players works
    @linq cc_friends = friends |>
    where(:country_index .== cc)
    @linq cc_top_game_info = top_game_info |>
    where(:country_index .== cc)
    # A has size (num_players * num_games)
    # now construct each (), set initial value equal to observed value:
    A_cc_game = zeros(num_games, cc_num_players) # the "2+" part represents the column that indicates the index and appid of game
    A_cc_friend = zeros(cc_num_players, cc_num_players)
    for ii in 1:cc_num_players
        println("gen play, friend info, conutry $cc, player $ii")
        # step 1: A_cc_game
        appid_i = cc_plays[ii,:appid] # get appid that i plays
        steamid_i = cc_plays[ii,:steamid] ## 3 is the column of steamid, this line gets the ID for player ii
        row_of_game= findall(cc_top_game_info.appid .== appid_i)[1] # get row number when appid is appid_i
        a_i = cc_plays[ii, :playtime_2weeks] # appid_i's playtime in the past 2 weeks
        # plug in a_i to the A matrix of game choice
        A_cc_game[row_of_game,ii] = (a_i > 0)
        # start to plug in g_i to A matrix of friends
        # step 2: A_cc_friend
        # a list of i's friends
        @linq i_friends = cc_friends |>
        where(:steamid_a .== steamid_i)
        num_i_friends = nrow(i_friends)
        for jj in 1:num_i_friends
            steamid_j = i_friends[jj, :steamid_b]
            column_of_friend = findall(cc_plays.steamid .== steamid_j)[1] # the column of the friend represented in the population matrix
            A_cc_friend[ii,column_of_friend] = 1
        end
    end
    return A_cc_game,A_cc_friend, cc_num_players, cc_top_game_info 
end
#@time (A_cc_game,A_cc_friend,cc_num_players,cc_top_game_info) = gen_cc(cc,top_game_info,num_games,friends,plays)

## 2.2 gen info about common group
function gen_cc_group(cc, groups, plays)
    @linq cc_plays = plays |>
    where(:country_index .== cc)
    cc_num_players = nrow(cc_plays) # one game per player, so num_players works
    @linq cc_groups = groups |>
    where(:country_index .== cc)
    A_cc_group = zeros(cc_num_players,cc_num_players)
    group_id_set = unique(cc_groups.groupid) # find groups appeared in cc
    cc_total_group = length(group_id_set)
    gg=1
    for gid in group_id_set
        println("gen group info, country $cc, group $gg, total group $(cc_total_group)")
        gg +=1
        @linq cc_groups_gid = cc_groups |>
        where(:groupid .== gid)  # find set of users in group gid
        steamid_with_gid = cc_groups_gid.steamid # find steamid's for this set of users
        steamid_row_number = findall(x -> x in steamid_with_gid, cc_plays.steamid) # locate these users row number
        A_cc_group[steamid_row_number,steamid_row_number] .+= 1.0 # add 1 to joint index in the shared group matrix 
    end
             # remove the trivial diagonal entries
    for ii in 1:cc_num_players
        A_cc_group[ii,ii] = 0
    end
    return A_cc_group
end
#@time A_cc_group = gen_cc_group(cc, groups, plays)



## 2.3 gen Dictionaries including all countries info
A_game = Dict{String, Matrix{Float64}}()
A_friend = Dict{String, Matrix{Float64}}()
A_num_players = Dict{String, Float64}()
A_top_game_info =  Dict{String, DataFrame}()
A_group = Dict{String, Matrix{Float64}}()
A_joint_play = Dict{String, Matrix{Float64}}()

A_row_is_mul =  Dict{String, Matrix{Float64}}()

#= 
##2.4 archived codes for gen dictionaries and save them
## 2.4.1 gen dictionaries for game, friend, num_players, top_game_info, group
for cc in 1:num_cc
    @time (A_game[Country_List[cc]],A_friend[Country_List[cc]],A_num_players[Country_List[cc]],A_top_game_info[Country_List[cc]]) = gen_cc(cc,top_game_info,num_games,friends,plays)
    @time A_group[Country_List[cc]] = gen_cc_group(cc, groups, plays)
end
# gen joint_play variable appeared in following utility_sum function 


## 2.4.2 gen dictionaries for joint_play
for cc in 1:num_cc
    joint_play_ij = zeros(Int64(A_num_players[Country_List[cc]]), Int64(A_num_players[Country_List[cc]]))
    for ii in 1:Int(A_num_players[Country_List[cc]])
        row_of_i_game = findall(A_game[Country_List[cc]][:, ii] .== 1.0)[1] # row of i's game in A game matrix, i.e. the game index played by player i   
        for jj in Int(ii+1):Int(A_num_players[Country_List[cc]])
            println("gen joint_play matrix for country $cc, ii = $ii, jj = $jj")
            joint_play_ij[ii,jj] = A_friend[Country_List[cc]][ii,jj] * A_game[Country_List[cc]][row_of_i_game, jj]
            joint_play_ij[jj,ii] = joint_play_ij[ii,jj]
        end
    end
    A_joint_play[Country_List[cc]] = joint_play_ij
end

## 2.4.3 gen dictionary for is multiplayer 
# and gen row of i game and whether game is multiplayer
for cc in 1:num_cc
    println("gen A_row_is_mul for country $cc")
    row_mul = zeros(Int64(A_num_players[Country_List[cc]]), 2) # 2: row_of_i_game & is_mul
    for ii in 1:Int(A_num_players[Country_List[cc]])
        row_of_i_game = findall(A_game[Country_List[cc]][:, ii] .== 1.0)[1] # row of i's game in A game matrix, i.e. the game index played by player i   
        is_mul = A_top_game_info[Country_List[cc]][row_of_i_game, :Is_Multiplayer]
        row_mul[ii, :] = vcat(row_of_i_game,is_mul)
    end
    A_row_is_mul[Country_List[cc]] = row_mul
end

## 2.4.4 save in txt files
for cc in 1:num_cc
    println("$cc")
    DelimitedFiles.writedlm("$(cc)_game.txt",  A_game[Country_List[cc]])
    DelimitedFiles.writedlm("$(cc)_friend.txt",  A_friend[Country_List[cc]])
    DelimitedFiles.writedlm("$(cc)_num_players.txt",  A_num_players[Country_List[cc]])
    CSV.write("$(cc)_top_game_info.csv",  A_top_game_info[Country_List[cc]])
    DelimitedFiles.writedlm("$(cc)_group.txt",  A_group[Country_List[cc]])
    DelimitedFiles.writedlm("$(cc)_joint_play.txt",  A_joint_play[Country_List[cc]])
    DelimitedFiles.writedlm("$(cc)_row_is_mul.txt",  A_row_is_mul[Country_List[cc]])
end
=#


## 2.4.5 read dictionaries
for cc in 1: num_cc
    println("cc = $cc")
    A_game[Country_List[cc]] = readdlm("$(cc)_game.txt")
    A_friend[Country_List[cc]] = readdlm("$(cc)_friend.txt")
    A_num_players[Country_List[cc]] = readdlm("$(cc)_num_players.txt")[1,1]
    A_top_game_info[Country_List[cc]] = CSV.read("$(cc)_top_game_info.csv", DataFrame)
    A_group[Country_List[cc]] = readdlm("$(cc)_group.txt")
    A_joint_play[Country_List[cc]] = readdlm("$(cc)_joint_play.txt")
    A_row_is_mul[Country_List[cc]] = readdlm("$(cc)_row_is_mul.txt")
end
#=
for cc in 1:num_cc
    println("cc = $cc")
    A_top_game_info[Country_List[cc]] = CSV.read("$(cc)_top_game_info.csv", DataFrame)
end
=#
## 2.4.6 (archive) save in jls files
# export these dictionary to local
#using Serialization

# Serialize the dictionary to a file
#filename = "A_game.jls"
#open(filename, "w") do io
#    serialize(io, A_game)
#end

#filename = "A_friend.jls"
#open(filename, "w") do io
#    serialize(io, A_friend)
#end

# gen an alternative JSON file for the partitioning implemntation in python
#filename = "A_friend.json"
#open(filename, "w") do io
#    JSON.print(io, A_friend)
#end
# yet another tractable file for the partitioning implemntation in python
#for cc in 1:num_cc
#    DelimitedFiles.writedlm("A_$(cc)_friend.txt", A_friend[Country_List[cc]])
#end


#filename = "A_num_players.jls"
#open(filename, "w") do io
#    serialize(io, A_num_players)
#end

#filename = "A_top_game_info.jls"
#open(filename, "w") do io
#    serialize(io, A_top_game_info)
#end

#filename = "A_group.jls"
#open(filename, "w") do io
#    serialize(io, A_group)
#end

#filename = "A_joint_play.jls"
#open(filename, "w") do io
#    serialize(io, A_joint_play)
#end



## 2.5 gen dataset for stata
# 2.5.1 gen game play dataset dta file
function gen_stata_frame(top_game_info,num_games,friends,plays)
    Stata_Frame = DataFrame(steamid = Any[], appid = Any[], country = Any[], i_play = Any[], num_friends_play = Any[], price = Any[], rating = Any[], age = Any[], is_mul = Any[],Release_Date = Any[], New_Rating = Any[], D_P_Match = Any[], Developer_Employee = Any[], Min_Ram = Any[], Max_Ram = Any[], Min_DirectX = Any[], Max_DirectX = Any[], Min_Space = Any[], Max_Space = Any[],num_ind_friends = Any[])
    new_row = Vector{Any}(zeros(20))# each row in stata_frame
    for cc in 1:num_cc
        new_row[3] = Country_List[cc]
        (A_cc_game,A_cc_friend,cc_num_players,cc_top_game_info) = gen_cc(cc,top_game_info,num_games,friends,plays)
        A_cc_connect = (A_cc_friend .>=1)
        cc_plays = filter(row -> row[:country_index] .== cc, plays)
        for ii in 1:cc_num_players
            ii_bool_vector = map(Bool, A_cc_connect[ii,:])
            new_row[1] = cc_plays[ii,:steamid] # steamid for ii
            new_row[20] = sum(A_cc_connect[ii_bool_vector,:]) ## ii's total number of friends
            for jj in 1:num_games
                println("importing data for country $cc, player $ii, game $jj")
                new_row[2] = cc_top_game_info[jj,:appid]
                new_row[4] = A_cc_game[jj,ii]
                new_row[5] = sum(A_cc_connect[ii,:] .* A_cc_game[jj,:])
                new_row[6] = cc_top_game_info[jj,:Price]
                new_row[7] = cc_top_game_info[jj,:Rating]
                new_row[8] = cc_top_game_info[jj,:Required_Age]
                new_row[9] = cc_top_game_info[jj,:Is_Multiplayer]
                new_row[10] = cc_top_game_info[jj,:Release_Date]
                new_row[11] = cc_top_game_info[jj,:New_Rating]
                new_row[12] = cc_top_game_info[jj,:D_P_Match]
                new_row[13] = cc_top_game_info[jj,:Developer_Employee]
                new_row[14] = cc_top_game_info[jj,:Min_Ram]
                new_row[15] = cc_top_game_info[jj,:Max_Ram]
                new_row[16] = cc_top_game_info[jj,:Min_DirectX]
                new_row[17] = cc_top_game_info[jj,:Max_DirectX]
                new_row[18] = cc_top_game_info[jj,:Min_Space]
                new_row[19] = cc_top_game_info[jj,:Max_Space]      
                push!(Stata_Frame, new_row)
            end
        end
    end
    CSV.write("/Users/shaoqiran/Documents/Documents - MacBook Pro/Rice University/JMP/Latest benchmark files/Stata files/Stata_Frame_new.csv", Stata_Frame)
end
#@time gen_stata_frame(A_top_game_info,num_games,friends,plays)

# 2.5.2 run regression for dyadic link formation
#=
using GLM
c_2 = vec(A_friend[Country_List[cc]])
c_3 = vec(A_group[Country_List[cc]])
# create a sample dataframe
df = DataFrame(
    X = c_3[1:1000000],
    Y = c_2[1:1000000]
)
probit = glm(@formula(Y ~ X), df, Binomial(), ProbitLink())
=#
#=
## 2.6 (archive) an artifical way to gen A_cc_friend, maximizing potential function
function gen_A_cc_friend_alter(A_cc_group, A_cc_total_game_time, cc_num_players, γ,ν)
    prob = ones(cc_num_players,cc_num_players) # prob that i,j are friends
    A_cc_friend = zeros(cc_num_players, cc_num_players)
    exp_term = zeros(cc_num_players, cc_num_players) # set storage
    for ii in 1:cc_num_players # 1:(cc_num_players-1)
        for jj in 1:cc_num_players #(ii+1):cc_num_players
            exp_term[ii,jj] = (exp(γ[1]
                            * abs((A_cc_total_game_time[ii]) - (A_cc_total_game_time[jj]))
                            +
                            γ[2]
                            * log(A_cc_group[ii,jj]+1) # +1 for normalization
                            +
                            γ[3]
                            * abs(ν[ii] - ν[jj])
                            )
                        )
            prob[ii,jj] = exp_term[ii,jj]/(1+exp_term[ii,jj])
            A_cc_friend[ii,jj] = (rand(1)[1] < prob[ii,jj])
        end
        A_cc_friend[ii,ii] = 0.0
    end
    return A_cc_friend
end
=#
#A_cc_friend = gen_A_cc_friend_alter(A_cc_group, A_cc_total_game_time, cc_num_players, γ,ν)



## 3 get summary statistics
#cc=1 # take cc = 1 for example

## 3.1 friend statistics
# statistics # of friends
#sum(A_cc_friend)
#mean(sum(A_cc_friend, dims = 1)[1,:])
#histogram(sum(A_cc_friend, dims = 1)[1,:])
#quantile!(sum(A_cc_friend, dims = 1)[1,:], 0.99)



## 3.2 player statistics

# # of players in total
#cc_num_players

# statistics of game play
#plot(sort(sum(A_cc_game, dims = 2)[:,1] ./cc_num_players))
# statistics # of friends
#histogram(sum(A_cc_connect, dims = 1)[1,:])
#quantile!(sum(A_cc_friend, dims = 1)[1,:], 0.99)
#A_cc_game
# number of groups in common
#A_cc_group_indicator = (A_cc_group .> 0)
#mean(sum(A_cc_group_indicator, dims = 1)[1,:])
#quantile!(sum(A_cc_group_indicator, dims = 1)[1,:], 0.99)
#histogram(sum(A_cc_group_indicator, dims = 1)[1,:])

# total game time
#mean(A_cc_total_game_time)
#quantile!(A_cc_total_game_time, 0.5)
#histogram(A_cc_total_game_time)
#minimum(A_cc_total_game_time)

## 3.3 game statistics


#print(top_game_info)
# price
#mean(top_game_info[:, :Price])
#quantile!(top_game_info[:, :Price], 0.95)
#histogram(top_game_info[:, :Price])

# rating
#mean(top_game_info[:, :Rating])
#quantile!(top_game_info[:, :Price], 0.95)
#histogram(top_game_info[:, :Price])

# required age
#mean(top_game_info[:, :Required_Age])
#quantile!(top_game_info[:, :Required_Age], 0.95)
#histogram(top_game_info[:, :Required_Age])

# is multiplayer
#mean(top_game_info[:, :Is_Multiplayer])
#quantile!(top_game_info[:, :Is_Multiplayer], 0.95)
#histogram(top_game_info[:, :Is_Multiplayer])



## 4 partition the graph
# 4.1 plot network graph
using Graphs
#A_cc_friend

#DelimitedFiles.writedlm("A_cc_friend.txt", A_cc_friend)
#g = Graph(A_cc_friend)
#sum(A_cc_friend)
#gplot(adj_matrix, nodefillc="white", nodeborderc="black", edgelabel=adj_matrix)
#print(g)
using GraphPlot
#@time gplot(g)

## 4.2 used python louvain algorigm to partition, now get a dictionary my_dict.json recording the partition map, use it to construct partioned adjacemcy matrix here
using JSON
num_partition = zeros(num_cc)
partition = Dict{String, Dict}()
for cc in 1:num_cc
    println("gen parition outcome for country $cc")
    # read the JSON file into a string
    json_string = read("A_$(cc)_friend_partitioned.json", String)

    # parse the JSON string into a dictionary
    my_dict = JSON.parse(json_string)

    # create a new dictionary with the same values but with the keys converted to Int
    my_dict = Dict{Int, Int}(parse(Int, key) => value for (key, value) in my_dict)

    # check number of partitions
    num_partition[cc] = length(Set(values(my_dict)))

    # community ids:
    partition_keys = 0:Int(num_partition[cc]-1)

    # outcome dictionary, each key is a partition, the corresponding values are the players indeces in the A_cc_friend matrix
    partition_outcome_cc = Dict(key => [] for key in partition_keys)

    for partition_id in Set(values(my_dict))
        # pick keys in the dictionary where the value is "New York"
        p_players = collect(keys(Dict(key => value for (key, value) in my_dict if value == partition_id))) .+ 1 # .+1 is crucial since python and julia use different indeces
        partition_outcome_cc[partition_id] = p_players
    end
    partition[Country_List[cc]] = partition_outcome_cc
end










## 4.3 plot subgraphs
#cc=1
#A_cc_frind_temp = A_friend[Country_List[cc]][partition_outcome[319], partition_outcome[319]]

#g = Graph(A_cc_frind_319)
#@time gplot(g)
#findmax([length(value) for value in values(partition_outcome_cc)])

#sorted_keys = sort(collect(keys(partition_outcome_cc)))
#for key in sorted_keys
#    println("$key => $(partition_outcome_cc[key])")
#end




## 4.4 compare g times a in full graph vs in subgraphs, if not much of difference, the partition does not affect likelihood estimator
#=
full_net_joint_play = zeros(num_cc)
sub_net_joint_play = zeros(num_cc)
full_net_link = zeros(num_cc)
sub_net_link = zeros(num_cc)
for cc in 1:num_cc
    # full graph joint_play in country cc
    full_net_joint_play[cc] = sum(A_joint_play[Country_List[cc]])
    # partitioned graph joint_play in country cc
    sub_net_joint_play[cc] = sum(sum(A_joint_play[Country_List[cc]][partition[Country_List[cc]][key], partition[Country_List[cc]][key]]) for key in keys(partition[Country_List[cc]]))
    #full graph links in country cc
    full_net_link[cc] = sum(A_friend[Country_List[cc]])
    # partitioned graph links in country cc
    sub_net_link[cc] = sum(sum(A_friend[Country_List[cc]][partition[Country_List[cc]][key], partition[Country_List[cc]][key]]) for key in keys(partition[Country_List[cc]]))
end

plot(full_net_joint_play)
plot!(sub_net_joint_play)
plot(full_net_link)
plot!(sub_net_link)
=#
## 5 try the following parameter values as benchmark
#=
τ_η = 1.0
δ = 0.5 # the coefficient in the price endogenity equation

β =  [1, 0.26, 0.35, 0.28]# constant, multiplayer, rating,  age

α = 0.2
ω = vcat(β, α)

num_rc = 3 # 3 random coefficients  on multiplayer, age, price
ρ = ones(num_rc+1) ./ 10 # representing variance term in ρ_ν, ρ_ξ
ϕ = [0.1, 0.2] # representing ϕ_s, ϕ_m

ξ = rand(Normal(0,1), (num_cc, num_games))


ν = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν[Country_List[cc]] =  rand(Normal(0,1), (Int(A_num_players[Country_List[cc]]), num_rc)) 
end
γ = vcat([-3.33,0.21], 0.1 * ones(num_rc))

σ_ηξ = 0.5 # need to be constrained so that σ_ηξ^2 < τ_η

cc=1

=#


## 6 from A_cc... to utility sum
# 6.1 normalize coefficients--- same as Stata 
#A_top_game_info_normalized = deepcopy(A_top_game_info)
for cc in 1:num_cc
    println("now normalizing price, age, rating variables for games in country $cc")
    #A_top_game_info[Country_List[cc]][:, :Price] = (A_top_game_info[Country_List[cc]][:, :Price])
    #A_top_game_info[Country_List[cc]][:, :Required_Age] = (A_top_game_info[Country_List[cc]][:, :Required_Age] .> 0)
    #A_top_game_info[Country_List[cc]][:, :New_Rating] = (A_top_game_info[Country_List[cc]][:, :New_Rating]) #new rating is already taken log
    A_top_game_info[Country_List[cc]][:, :Metascore] = deepcopy(log.((A_top_game_info[Country_List[cc]][:, :Rating])))
    A_top_game_info[Country_List[cc]][:, :Developer_Employee] = deepcopy(log.(A_top_game_info[Country_List[cc]][:, :Developer_Employee] .+1.01))
end



## define data structure

struct Steam_struct_primary
    game_play::Dict{String, Matrix{Float64}}
    friend::Dict{String, Matrix{Float64}}
    num_players::Dict{String, Float64}
    top_game_info::Dict{String, DataFrame}
    group::Dict{String, Matrix{Float64}}
    joint_play::Dict{String, Matrix{Float64}}
    row_is_mul::Dict{String, Matrix{Float64}}
    partition_index::Dict{String, Dict}
end



#length(m.game_play[Country_List[1]][1,:])
function Steam_model(A_game, A_friend, A_num_players, A_top_game_info, A_group, A_joint_play, A_row_is_mul, partition)
    game_play = (A_game)
    friend = (A_friend)
    num_players = (A_num_players)
    top_game_info = (A_top_game_info)
    group = (A_group)
    joint_play = (A_joint_play)
    row_is_mul = (A_row_is_mul)
    partition_index = (partition)
    model = Steam_struct_primary(game_play, friend, num_players, top_game_info, group, joint_play, row_is_mul, partition_index)
    return model
end
m = Steam_model(A_game, A_friend, A_num_players, A_top_game_info, A_group, A_joint_play, A_row_is_mul, partition)

#print(m.top_game_info[Country_List[cc]])

# normalize price by firm size
for cc in 1:num_cc
    m.top_game_info[Country_List[cc]][:,:Price] = m.top_game_info[Country_List[cc]][:,:Price] ./ A_top_game_info[Country_List[cc]][:, :Min_Ram]
end


#plot(m.top_game_info[Country_List[3]][:,:Price] ./A_top_game_info[Country_List[cc]][:, :Min_Ram])
#plot(m.top_game_info[Country_List[10]][:,:Price])

#plot(A_top_game_info[Country_List[cc]][:, :Max_Ram])
#print(m.top_game_info[Country_List[3]])



#sum(m.friend["DE"], dims = 1)[1,:]
#histogram(sum(m.friend["DE"], dims = 1)[1,:])

#cc=2;key=127
function gen_A_utility_sum(cc, key, m, β, α, ρ, ν, ϕ, ξ)
    player_rows = m.partition_index[Country_List[cc]][key]
    cc_top_game_info = m.top_game_info[Country_List[cc]]
    A_cc_game = m.game_play[Country_List[cc]][:,player_rows]
    A_cc_friend = m.friend[Country_List[cc]][player_rows, player_rows]
    A_cc_row_is_mul = m.row_is_mul[Country_List[cc]][player_rows,:]
    A_cc_joint_play = m.joint_play[Country_List[cc]][player_rows, player_rows]
    
    utility_i = zeros(length(player_rows)) # number of players in this partition in cc is length(player_rows)
    for ii in 1:length(player_rows)
        ##println("ii=$ii")
        player_i_row = player_rows[ii]
        row_of_i_game = Int(A_cc_row_is_mul[ii,1])
        is_i_game_mul = A_cc_row_is_mul[ii,2]# row of i's game in A game matrix, i.e. the game index played by player i
        utility_i[ii] = (β[1] + # rating, multiplayer, age
                        + (cc_top_game_info[row_of_i_game, :Is_Multiplayer]) * (β[2] + ρ[1] * ν[Country_List[cc]][player_i_row,1]) # Is_Multiplayer
                        + (cc_top_game_info[row_of_i_game, :New_Rating]) * β[3] # rating, no random coefficient
                        + cc_top_game_info[row_of_i_game, :Required_Age] * (β[4] + ρ[2] * ν[Country_List[cc]][player_i_row,2]) # age
                        - (cc_top_game_info[row_of_i_game, :Price]) * (α + ρ[3] * ν[Country_List[cc]][player_i_row,3]) # price
                        + ρ[4] * ξ[cc, row_of_i_game] # game fixed effect
                        + 1/2 * ϕ[1] * (1-is_i_game_mul) * sum(A_cc_joint_play[ii,:]) # if single player
                        + 1/2 * ϕ[2] * is_i_game_mul * sum(A_cc_joint_play[ii,:])) # term for network effect of multiplayer game
    end
    utility_sum = sum(utility_i)
    return utility_sum
end

#@time utility_sum = gen_A_utility_sum(cc, key, m, β, α, ρ, ν, ϕ, ξ)
#=
plot(m.top_game_info[Country_List[cc]][:,:Min_Ram])
plot(m.top_game_info[Country_List[cc]][:,:Price])
=#
#x=m.top_game_info[Country_List[cc]][:,:Min_Ram]
#y=m.top_game_info[Country_List[cc]][:,:Price]

#x\y

#cc=1;key = 1144 
function gen_A_R_utility_sum(cc, key, A_R, m, β, α, ρ, ν, ϕ, ξ) # need to regen variables in this case
    #ν = Dict{String, Matrix{Float64}}()
    #for cc in 1:num_cc
    #    ν[Country_List[cc]] =  rand(Normal(0,1), (Int(A_num_players[Country_List[cc]]), num_rc)) 
    #end
    #ξ = rand(Normal(0,1), (num_cc, num_games))
    A_cc_num_players = length(m.partition_index[Country_List[cc]][key])
    player_rows = m.partition_index[Country_List[cc]][key]
    cc_top_game_info = m.top_game_info[Country_List[cc]]
    A_cc_friend = m.friend[Country_List[cc]][player_rows, player_rows]

    A_cc_game = copy(A_R)

    utility_i = zeros(A_cc_num_players) # number of players in this partition in cc is length(player_rows)
    for ii in 1:A_cc_num_players
        row_of_i_game = findall(A_cc_game[:, ii] .== 1.0)[1] # row of i's game in A game matrix, i.e. the game index played by player i   
        player_i_row = player_rows[ii]
        is_i_game_mul = cc_top_game_info[row_of_i_game, :Is_Multiplayer] # row of i's game in A game matrix, i.e. the game index played by player i
        utility_i[ii] = (β[1] + # rating, multiplayer, age
                        + (cc_top_game_info[row_of_i_game, :Is_Multiplayer]) * (β[2] + ρ[1] * ν[Country_List[cc]][player_i_row,1]) # Is_Multiplayer
                        + (cc_top_game_info[row_of_i_game, :New_Rating]) * β[3] # rating, no random coefficient
                        + cc_top_game_info[row_of_i_game, :Required_Age] * (β[4] + ρ[2] * ν[Country_List[cc]][player_i_row,2]) # age
                        -  (cc_top_game_info[row_of_i_game, :Price]) * (α + ρ[3] * ν[Country_List[cc]][player_i_row,3]) # price
                        + ρ[4] * ξ[cc, row_of_i_game]  # game fixed effect
                        + 1/2 * ϕ[1] * (1-is_i_game_mul) * sum(A_cc_friend[ii,:] .* A_cc_game[row_of_i_game, :]) # if single player
                        + 1/2 * ϕ[2] * is_i_game_mul * sum(A_cc_friend[ii,:] .* A_cc_game[row_of_i_game, :])) # term for network effect of multiplayer game
    end
    utility_sum = sum(utility_i)
    return utility_sum
end
 # for testing the validity of function gen_A_R_utility_sum(): A_R = A_cc_game then results of two functions should equal
#=
 player_rows = m.partition_index[Country_List[cc]][key]
cc_top_game_info = m.top_game_info[Country_List[cc]]
A_cc_game = m.game_play[Country_List[cc]][:,player_rows]
A_R = copy(A_cc_game)
@time utility_sum = gen_A_R_utility_sum(cc, key, A_R, m, β, α, ρ, ν, ϕ, ξ)
=#
# check utility_sum if all play same game
#=
utility_sum = zeros(num_games)
for gg in 1:num_games
    A_R = zeros(num_games,79)
    A_R[gg,:] = ones(79)
    @time utility_sum[gg] = gen_A_R_utility_sum(2, 127, A_R, m, β, α, ρ, ν, ϕ, ξ)
end

findmax(utility_sum)
=#
## 6.2 MC algorithm to generate sample of fake A's for each cc, test the validity of proposition 1
#R = 10
#cc=1;key=1144
function MC_A(R,cc, key, m, β, α, ρ, ν, ϕ, ξ) # rate: how attractive game number 1 is
    player_rows = m.partition_index[Country_List[cc]][key]
    num_games = nrow(m.top_game_info[Country_List[cc]])
    A_cc_num_players =  length(m.partition_index[Country_List[cc]][key]) 
    A_cc_game = m.game_play[Country_List[cc]][:,player_rows] # observed A_cc_game in partitioned sample
    #A_cc_friend = m.friend[Country_List[cc]][player_rows, player_rows]
    #A_cc_joint_play =  m.joint_play[Country_List[cc]][player_rows, player_rows]
    A_cc_game_MC_sample = zeros(R,num_games, A_cc_num_players)
    #A_cc_row_is_mul = m.row_is_mul[Country_List[cc]][player_rows,:] # 2 represents: row_of_i_game & is_mul
    utility_sum_prev = gen_A_utility_sum(cc, key, m, β, α, ρ, ν, ϕ, ξ) #for comparing temp with prev in MC
    for rr in 1:R
        #println("rr = $rr")
        #=
        decision_maker = sample(1:A_cc_num_players)
        utility_temp = zeros(num_games)
        for gg in 1:num_games
            A_cc_game[:,decision_maker] = zeros(num_games)
            A_cc_game[gg, decision_maker] = 1.0
                row_of_i_game = findall(A_cc_game[:, decision_maker] .== 1.0)[1] # row of i's game in A game matrix, i.e. the game index played by player i   
                is_i_game_mul  = cc_top_game_info[row_of_i_game, :Is_Multiplayer]
                A_cc_row_is_mul[decision_maker,:] = vcat(row_of_i_game, is_mul) 
                for jj in 1:A_cc_num_players
                    println("gen joint_play matrix for country $cc, jj = $jj")
                    A_cc_joint_play[decision_maker,jj] = A_cc_friend[decision_maker,jj] * A_cc_game[row_of_i_game, jj]
                end

            utility_temp[gg] = (β[1] + # rating, multiplayer, age
                                + (cc_top_game_info[gg, :Is_Multiplayer]) * (β[2] + ρ[1] * ν[Country_List[cc]][player_rows[decision_maker],1]) # Is_Multiplayer
                                + (cc_top_game_info[gg, :New_Rating]) * β[3] # rating, no random coefficient
                                + cc_top_game_info[gg, :Required_Age] * (β[4] + ρ[2] * ν[Country_List[cc]][player_rows[decision_maker],2]) # age
                                -  (cc_top_game_info[gg, :Price]) * (α + ρ[3] * ν[Country_List[cc]][player_rows[decision_maker],3]) # price
                                + ρ[4] * ξ[gg] # game fixed effect
                                + 1/2 * ϕ[1] * (1-is_i_game_mul) * sum(A_cc_joint_play[decision_maker,:]) # if single player
                                + 1/2 * ϕ[2] * is_i_game_mul * sum(A_cc_joint_play[decision_maker,:]) # term for network effect of multiplayer game
                                + rand(GeneralizedExtremeValue(0,1,0)))
        end
        A_cc_game[:,decision_maker] = zeros(num_games)
        A_cc_game[findmax(utility_temp)[2], decision_maker] = 1.0
        A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game)

        =#

        
         # draw k: 75% prob. k =2; 25% k is discrete uniform on 2, ..., cc_num_players - 1
        rate_temp = rand(1)[1]
        if rate_temp < 0.75 # the 75% case
            k = min(rand(Poisson(4)) + 1, A_cc_num_players)
        else
            k = A_cc_num_players
        end
        #k = min(rand(Poisson(4)) + 1, cc_num_players)
        set_i = sample(1:A_cc_num_players, k) # select set of decision makers
        #replace action
        A_cc_game_temp = copy(A_cc_game)
        for ii in set_i
            A_cc_game_temp[:, ii] = zeros(num_games)
            game = rand(1:num_games) # game to be played as an alternative
            A_cc_game_temp[game, ii] = 1.0 # force to play this game instead
        end
        # method 2 of drawing k:
        # draw k: 75% prob. k =2; 25% k is discrete uniform on 2, ..., cc_num_players - 1
        #rate_temp = rand(1)[1]
        #if rate_temp < 0.75 # the 75% case
        #    k = 2
        #else
        #    k = sample(2: (cc_num_players - 1))
        #end
        #set_i = sample(1:cc_num_players, k) # select set of decision makers
        #replace action
        #A_cc_game_temp = copy(A_cc_game)
        #for ii in set_i
        #    A_cc_game_temp[:, ii] = zeros(num_games)
        #    game = rand(1:num_games) # game to be played as an alternative
        #    A_cc_game_temp[game, ii] = 1.0 # force to play this game instead
        #end
        utility_sum_temp = gen_A_R_utility_sum(cc, key, A_cc_game_temp, m, β, α, ρ, ν, ϕ, ξ)
        a_bar = utility_sum_temp - utility_sum_prev # a_bar, remove exp since it explodes, the bar is logged version of the original model
        a_rate = log(rand(1)[1])
        if a_rate >= a_bar
            A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game)
        else # plug in new A,
            A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game_temp)
            A_cc_game = copy(A_cc_game_temp)
            utility_sum_prev = copy(utility_sum_temp)
        end
        
    end
    return A_cc_game_MC_sample[R,:,:]
end
#@time A_R = MC_A(R,cc, key, m, β, α, ρ, ν, ϕ, ξ) 



# MC for full country
function MC_A_cc_full(R,cc, m, β, α, ρ, ν, ϕ, ξ) # rate: how attractive game number 1 is
    num_games = nrow(m.top_game_info[Country_List[cc]])
    A_cc_num_players =  length(m.game_play[Country_List[1]][1,:])
    A_cc_game = m.game_play[Country_List[cc]] # observed A_cc_game in partitioned sample
    #A_cc_friend = m.friend[Country_List[cc]][player_rows, player_rows]
    #A_cc_joint_play =  m.joint_play[Country_List[cc]][player_rows, player_rows]
    A_cc_game_MC_sample = zeros(R,num_games, A_cc_num_players)
    #A_cc_row_is_mul = m.row_is_mul[Country_List[cc]][player_rows,:] # 2 represents: row_of_i_game & is_mul
    utility_sum_prev = gen_A_utility_sum(cc, key, m, β, α, ρ, ν, ϕ, ξ) #for comparing temp with prev in MC
    for rr in 1:R
        println("cc = $cc, rr = $rr")
        #=
        decision_maker = sample(1:A_cc_num_players)
        utility_temp = zeros(num_games)
        for gg in 1:num_games
            A_cc_game[:,decision_maker] = zeros(num_games)
            A_cc_game[gg, decision_maker] = 1.0
                row_of_i_game = findall(A_cc_game[:, decision_maker] .== 1.0)[1] # row of i's game in A game matrix, i.e. the game index played by player i   
                is_i_game_mul  = cc_top_game_info[row_of_i_game, :Is_Multiplayer]
                A_cc_row_is_mul[decision_maker,:] = vcat(row_of_i_game, is_mul) 
                for jj in 1:A_cc_num_players
                    println("gen joint_play matrix for country $cc, jj = $jj")
                    A_cc_joint_play[decision_maker,jj] = A_cc_friend[decision_maker,jj] * A_cc_game[row_of_i_game, jj]
                end

            utility_temp[gg] = (β[1] + # rating, multiplayer, age
                                + (cc_top_game_info[gg, :Is_Multiplayer]) * (β[2] + ρ[1] * ν[Country_List[cc]][player_rows[decision_maker],1]) # Is_Multiplayer
                                + (cc_top_game_info[gg, :New_Rating]) * β[3] # rating, no random coefficient
                                + cc_top_game_info[gg, :Required_Age] * (β[4] + ρ[2] * ν[Country_List[cc]][player_rows[decision_maker],2]) # age
                                -  (cc_top_game_info[gg, :Price]) * (α + ρ[3] * ν[Country_List[cc]][player_rows[decision_maker],3]) # price
                                + ρ[4] * ξ[gg] # game fixed effect
                                + 1/2 * ϕ[1] * (1-is_i_game_mul) * sum(A_cc_joint_play[decision_maker,:]) # if single player
                                + 1/2 * ϕ[2] * is_i_game_mul * sum(A_cc_joint_play[decision_maker,:]) # term for network effect of multiplayer game
                                + rand(GeneralizedExtremeValue(0,1,0)))
        end
        A_cc_game[:,decision_maker] = zeros(num_games)
        A_cc_game[findmax(utility_temp)[2], decision_maker] = 1.0
        A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game)

        =#

        
         # draw k: 75% prob. k =2; 25% k is discrete uniform on 2, ..., cc_num_players - 1
        rate_temp = rand(1)[1]
        if rate_temp < 0.75 # the 75% case
            k = min(rand(Poisson(4)) + 1, A_cc_num_players)
        else
            k = A_cc_num_players
        end
        #k = min(rand(Poisson(4)) + 1, cc_num_players)
        set_i = sample(1:A_cc_num_players, k) # select set of decision makers
        #replace action
        A_cc_game_temp = copy(A_cc_game)
        for ii in set_i
            A_cc_game_temp[:, ii] = zeros(num_games)
            game = rand(1:num_games) # game to be played as an alternative
            A_cc_game_temp[game, ii] = 1.0 # force to play this game instead
        end
        # method 2 of drawing k:
        # draw k: 75% prob. k =2; 25% k is discrete uniform on 2, ..., cc_num_players - 1
        #rate_temp = rand(1)[1]
        #if rate_temp < 0.75 # the 75% case
        #    k = 2
        #else
        #    k = sample(2: (cc_num_players - 1))
        #end
        #set_i = sample(1:cc_num_players, k) # select set of decision makers
        #replace action
        #A_cc_game_temp = copy(A_cc_game)
        #for ii in set_i
        #    A_cc_game_temp[:, ii] = zeros(num_games)
        #    game = rand(1:num_games) # game to be played as an alternative
        #    A_cc_game_temp[game, ii] = 1.0 # force to play this game instead
        #end
        utility_sum_temp = gen_A_R_utility_sum(cc, key, A_cc_game_temp, m, β, α, ρ, ν, ϕ, ξ)
        a_bar = utility_sum_temp - utility_sum_prev # a_bar, remove exp since it explodes, the bar is logged version of the original model
        a_rate = log(rand(1)[1])
        if a_rate >= a_bar
            A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game)
        else # plug in new A,
            A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game_temp)
            A_cc_game = copy(A_cc_game_temp)
            utility_sum_prev = copy(utility_sum_temp)
        end
        
    end
    return A_cc_game_MC_sample[R,:,:]
end
#@time A_R = MC_A(R,cc, key, m, β, α, ρ, ν, ϕ, ξ) 

#=

R=1000
game_benchmark = 12
rate = 0.5
cc=2;key=127
A_cc_game = m.game_play[Country_List[2]]
=#
function MC_A_density(R,cc, key, m, β, α, ρ, ν, ϕ, ξ,game_benchmark, rate) # rate: how attractive game number 1 is
    #A_cc_row_is_mul = zeros(A_cc_num_players, 2) # 2 represents: row_of_i_game & is_mul
    player_rows = m.partition_index[Country_List[cc]][key]
    num_games = nrow(m.top_game_info[Country_List[cc]])
    A_cc_num_players =  length(m.partition_index[Country_List[cc]][key])
    A_cc_game_MC_sample = zeros(R,num_games, A_cc_num_players)
    A_cc_game_density = zeros(R, num_games)
    A_cc_game_dispersion = zeros(R)
    w = ones(num_games)
    for gg in 1:num_games
        if gg == game_benchmark
            w[gg] = rate
        else
            w[gg] = (1 - rate)/num_games
        end
    end
    A_cc_game=zeros(num_games, A_cc_num_players)
    for ii in 1:A_cc_num_players
        A_cc_game[sample(1:num_games, Weights(w)),ii] = 1.0
    end
    # start from observed data, for model fit test
    #player_rows = m.partition_index[Country_List[cc]][key]
    #num_games = nrow(m.top_game_info[Country_List[cc]])
    #A_cc_num_players =  length(m.partition_index[Country_List[cc]][key]) 
    A_cc_game = m.game_play[Country_List[cc]][:,player_rows] # observed A_cc_game in partitioned sample
    
    utility_sum_prev = gen_A_R_utility_sum(cc, key, A_cc_game, m, β, α, ρ, ν, ϕ, ξ) #for comparing temp with prev in MC

    for rr in 1:R
        #println("rr = $rr")

        #=
        decision_maker = sample(1:A_cc_num_players)
        is_i_game_mul = A_cc_row_is_mul[decision_maker,2]# row of i's game in A game matrix, i.e. the game index played by player i
        utility_temp = zeros(num_games)
        for gg in 1:num_games
            A_cc_game[:,decision_maker] = zeros(num_games)
            A_cc_game[gg, decision_maker] = 1.0
            utility_temp[gg] = (β[1] + # rating, multiplayer, age
                                + (cc_top_game_info[gg, :Is_Multiplayer]) * (β[2] + ρ[1] * ν[Country_List[cc]][player_rows[decision_maker],1]) # Is_Multiplayer
                                + (cc_top_game_info[gg, :New_Rating]) * β[3] # rating, no random coefficient
                                + cc_top_game_info[gg, :Required_Age] * (β[4] + ρ[2] * ν[Country_List[cc]][player_rows[decision_maker],2]) # age
                                -  (cc_top_game_info[gg, :Price]) * (α + ρ[3] * ν[Country_List[cc]][player_rows[decision_maker],3]) # price
                                + ρ[4] * ξ[gg] # game fixed effect
                                + 1/2 * ϕ[1] * (1-is_i_game_mul) * sum(A_cc_joint_play[decision_maker,:]) # if single player
                                + 1/2 * ϕ[2] * is_i_game_mul * sum(A_cc_joint_play[decision_maker,:]) # term for network effect of multiplayer game
                                + rand(GeneralizedExtremeValue(0,1,0)))






        end
        A_cc_game[:,decision_maker] = zeros(num_games)
        A_cc_game[findmax(utility_temp)[2], decision_maker] = 1.0
        A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game)
        
        =#
        
         # draw k: 75% prob. k =2; 25% k is discrete uniform on 2, ..., A_cc_num_players - 1
        rate_temp = rand(1)[1]
        if rate_temp < 0.75 # the 75% case
            k = min(rand(Poisson(4)) + 1, A_cc_num_players)
        else
            k = A_cc_num_players
        end
        #k = min(rand(Poisson(4)) + 1, A_cc_num_players)
        set_i = sample(1:A_cc_num_players, k) # select set of decision makers
        #replace action
        A_cc_game_temp = copy(A_cc_game)
        for ii in set_i
            A_cc_game_temp[:, ii] = zeros(num_games)
            game = rand(1:num_games) # game to be played as an alternative
            A_cc_game_temp[game, ii] = 1.0 # force to play this game instead
        end

        utility_sum_temp = gen_A_R_utility_sum(cc, key, A_cc_game_temp, m, β, α, ρ, ν, ϕ, ξ)

        a_bar = utility_sum_temp - utility_sum_prev # a_bar, remove exp since it explodes, the bar is logged version of the original model
        a_rate = log(rand(1)[1])
        if a_rate >= a_bar
            A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game)
        else # plug in new A,
            A_cc_game_MC_sample[rr,:,:] = copy(A_cc_game_temp)
            A_cc_game = copy(A_cc_game_temp)
            utility_sum_prev = copy(utility_sum_temp)
        end






        for gg in 1:num_games
            A_cc_game_density[rr, gg] = mean(A_cc_game[gg,:])

        end
        A_cc_game_dispersion[rr] = std(A_cc_game_density[rr, :])
    end
    return A_cc_game_density, A_cc_game_dispersion
end
#@time A_cc_game_density, A_cc_game_dispersion = MC_A_density(R,cc, key, m, β, α, ρ, ν, ϕ, ξ,game_benchmark, rate) 

#plot(A_cc_game_dispersion)
#plot(A_cc_game_density[:,12])




## 6.3 gen the likelihood for the network matrix
function gen_μ_and_μ_log_sum(cc, key, m, γ,ν)
    player_rows = m.partition_index[Country_List[cc]][key]
    A_cc_friend = m.friend[Country_List[cc]][player_rows, player_rows]
    A_cc_group = m.group[Country_List[cc]][player_rows, player_rows]
    cc_num_players = length(player_rows)

    μ=ones(cc_num_players,cc_num_players)
    exp_term = zeros(cc_num_players, cc_num_players) # set storage
    for ii in 1:cc_num_players #1:(cc_num_players-1)
        for jj in 1:cc_num_players #(ii+1):cc_num_players
            player_i_row = player_rows[ii]
            player_j_row = player_rows[jj]
            exp_term[ii,jj] = (
                            exp(γ[1]
                            +
                            γ[2]
                            * log(A_cc_group[ii,jj]+1) # +1 for normalization
                            +
                            sum(γ[2+ii] * 
                            abs(ν[Country_List[cc]][player_i_row,ii] - ν[Country_List[cc]][player_i_row,ii])
                            for ii in 1:num_rc
                            )
                            )
                        )
            μ[ii,jj] = (A_cc_friend[ii,jj] == 1) * exp_term[ii,jj]/(1+exp_term[ii,jj]) + (A_cc_friend[ii,jj] == 0) /(1+exp_term[ii,jj])
        end
        μ[ii,ii] = 1.0
    end
    μ_log_sum = sum(sum( log(μ[ii,jj]) for jj in 1:cc_num_players) for ii in 1:cc_num_players)
    return μ, μ_log_sum
end
#(μ, μ_log_sum) = gen_μ_and_μ_log_sum(cc, key, m, γ,ν)


#ii=40;jj=62
function gen_μ_log_sum(cc, key, m, γ,ν)
    player_rows = m.partition_index[Country_List[cc]][key]
    A_cc_friend = m.friend[Country_List[cc]][player_rows, player_rows]
    A_cc_group = m.group[Country_List[cc]][player_rows, player_rows]
    cc_num_players = length(player_rows)

    μ=ones(cc_num_players,cc_num_players)
    exp_term = zeros(cc_num_players, cc_num_players) # set storage
    for ii in 1:cc_num_players #1:(cc_num_players-1)
        for jj in 1:cc_num_players #(ii+1):cc_num_players
            player_i_row = player_rows[ii]
            player_j_row = player_rows[jj]
            exp_term[ii,jj] = (
                            exp(γ[1]
                            +
                            γ[2]
                            * log(A_cc_group[ii,jj]+1) # +1 for normalization
                            +
                            sum(γ[2+ii] * 
                            abs(ν[Country_List[cc]][player_i_row,ii] - ν[Country_List[cc]][player_i_row,ii])
                            for ii in 1:num_rc
                            )
                            )
                        )
            μ[ii,jj] = (A_cc_friend[ii,jj] == 1) * exp_term[ii,jj]/(1+exp_term[ii,jj]) + (A_cc_friend[ii,jj] == 0) /(1+exp_term[ii,jj])
        end
        μ[ii,ii] = 1.0
    end
    μ_log_sum = sum(sum( log(μ[ii,jj]) for jj in 1:cc_num_players) for ii in 1:cc_num_players)
    return μ_log_sum
end
#μ_log_sum = gen_μ_log_sum(A_cc_friend, A_cc_group, A_cc_total_game_time, cc_num_players, γ,ν)
#findmin(μ)
#ii = 694; jj = 557

## 7 set price and IV
price = Vector{Float64}(vcat([m.top_game_info[Country_List[cc]][:,:Price] for cc in 1:num_cc]...))
IV = vcat([m.top_game_info[Country_List[cc]][:, :Developer_Employee]  for cc in 1:num_cc]...)
IV = Matrix{Float64}(hcat(ones(num_games * num_cc), IV))


#D_E = vcat([m.top_game_info[Country_List[cc]][:, :Developer_Employee]  for cc in 1:num_cc]...)





#alternative IV
#using Dates
#=
query_date =DateTime("2014-08-14")

IV_temp = vcat([m.top_game_info[Country_List[cc]][:, :Release_Date]  for cc in 1:num_cc]...)
IV_time = [DateTime(release_time, "yyyy-mm-dd HH:MM:SS") for release_time in IV_temp]

IV = log.([Dates.value(query_date) - Dates.value(release_date) for release_date in IV_time])


IV = log.([Dates.value(query_date) - Dates.value(release_date) for release_date in IV_time])

IV = Matrix{Float64}(hcat(ones(num_games * num_cc), IV))

=#





## 8 specify the prior distribution (hyperparameters) of parameters
## 8.1 precision of error term in the price endogenity equation
α_τ_η = 2.0
β_τ_η = 2.0



## 8.2 coefficients of BLP instruments in the price endogeneity equation
μ_δ = Vector{Float64}(inv(IV' * IV) *  IV' * price) # price according to first stage OLS result
Σ_δ = [1.0 0.0; 0.0 1.0]



###### here!
## 8.3 coefficients of gaming characteristics and prices
k=4 # constant, multiplayer, rating,  age
β =  [1, 0.26, 0.35, 0.28]# constant, multiplayer, rating,  age
α = 0.2
μ_ω = vcat(β,α) # 5+1 =6
Σ_ω = Matrix(I, k+1, k+1)


## 8.4 coefficients of individual tastes on gaming and unobserved game quality
num_rc = 3 # 3 random coefficients  on multiplayer, age, price
μ_ρ = ones(num_rc+1) ./ 10 # representing variance term in ρ_ν, ρ_ξ
Σ_ρ = Matrix(I, num_rc+1, num_rc+1) # +1 since need to estimate variance on ξ

## 8.5 coefficients of peer effects
μ_ϕ = [0.1, 0.2]
Σ_ϕ = Matrix(I, 2, 2)

## 8.6 coefficients of dyadic formation process
μ_γ = vcat([-3.33,0.21], 0.1 * ones(num_rc))
Σ_γ = Matrix(I, num_rc+2, num_rc+2) # num of regressors in dyadic regression


## 8.7 coefficients of latent variable on gaming taste $\nu_i$ for each $i$
#N(0,1)

## 8.8 coefficients of latent variable on game quality $\xi_b$ for each $b$
μ_σ_ηξ = 0.5
Σ_σ_ηξ = 1.0



## 9 Detailed MCMC steps
T = 100000 # num of outer MH iterations
R = 2 # num of inner MH
n_par = 20 # when simulating A_R, number of partition used
## 9.1 set up MCMC draws memory for each parameters
τ_η_MC = zeros(T)
δ_MC = zeros(T,2) # the coefficient in the price endogenity equation

β_MC = zeros(T,k) # constant, multiplayer, rating,  age
α_MC = zeros(T)
ω_MC = zeros(T, k+1)

ρ_MC = zeros(T,num_rc+1)  # representing variance term in ρ_ν, ρ_ξ
ϕ_MC = zeros(T,2) # representing ϕ_s, ϕ_m
γ_MC = zeros(T,num_rc+2) # constant, num_groups in common, plus number of random coefficient

ν_MC = Dict{Int64,  Dict{String, Matrix{Float64}}}()
ν_MC_tt =  Dict{String, Matrix{Float64}}()
ξ_MC = zeros(T,num_cc, num_games)

σ_ηξ_MC = zeros(T) # need to be constrained so that σ_ηξ^2 < τ_η

η_MC = zeros(T, num_cc, num_games) # computed during MCMC
## 9.2 detailed MCMC samples
## initial draw
tt=1
τ_η_MC[tt] = rand(Gamma(α_τ_η, 1/β_τ_η))
δ_MC[tt,:] = rand(Distributions.MvNormal(μ_δ, τ_η_MC[tt]^(-1) * Σ_δ)) # the coefficient in the price endogenity equation

ω_MC[tt,:] = rand(Distributions.MvNormal(μ_ω, Σ_ω)) # the coefficient in the main utility v_ib

β_MC[tt,:] = ω_MC[tt,1:k] # constant, multiplayer, rating,  age
α_MC[tt] = ω_MC[tt,k+1]

ω_MC[1,:]

ρ_MC[tt,:] = rand(Distributions.MvNormal(μ_ρ, Σ_ρ))
ϕ_MC[tt,:] = rand(Distributions.MvNormal(μ_ϕ, Σ_ϕ)) # representing ϕ_s, ϕ_m
γ_MC[tt,:] = rand(Distributions.MvNormal(μ_γ, Σ_γ)) # constant,  diff in total_game_time, num_groups in common,



for cc in 1:num_cc
    ν_MC_tt[Country_List[cc]] = rand(Normal(0,1), (Int(A_num_players[Country_List[cc]]), num_rc)) 
end
ν_MC[tt] =  ν_MC_tt

#ξ_MC[tt,:,:] = rand(Normal(0,1), (num_cc, num_games))

σ_ηξ_MC[tt] = rand(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC[tt]^(-1/2)),  τ_η_MC[tt]^(-1/2))) # need to be constrained so that σ_ηξ^2 < τ_η



η_MC[tt,:,:] = reshape(price - IV * δ_MC[tt,:], (num_games,num_cc))'

#ξ_MC[tt,:,:] = rand(Normal(0,1), (num_cc, num_games))

for cc in 1:num_cc
    for bb in 1:num_games
        ξ_MC[tt,cc,bb] = rand(Normal(σ_ηξ_MC[tt] * τ_η_MC[tt] * η_MC[tt,cc,bb],sqrt(1- σ_ηξ_MC[tt]^2 * τ_η_MC[tt])))
    end
end


## for tt >= 2, initiate the MCMC:
# some preparations
# in 1
n_z = 2  # n_z is the number of regressors in price endogeneity regression
# in 2
Q_δ = IV' * IV + inv(Σ_δ)
inv_Q_δ =  inv(Q_δ)
l_δ = IV' * price + inv(Σ_δ) * μ_δ
inv_Q_δ_l_δ = inv(Q_δ) * l_δ

# in 3(a)
shrink = 5
proposal_var_ω = diagm(μ_ω) * Σ_ω ./ shrink # 5 is tuning parameter for the proposal distribution, same below
proposal_mean_ω = zeros(k+1)

# in 4(a)
proposal_var_ρ = diagm(μ_ρ) * Σ_ρ ./ shrink
proposal_mean_ρ = zeros(num_rc+1)

# in 5(a)
proposal_var_ϕ = diagm(μ_ϕ) * Σ_ϕ ./ shrink
proposal_mean_ϕ = zeros(2)

# in 6(a)
proposal_var_γ = Σ_γ ./ shrink
proposal_mean_γ = zeros(num_rc+2)

# in 7
#exp_term_lag = zeros(cc_num_players)
#μ_lag = zeros(cc_num_players)
#exp_term_tilde = zeros(cc_num_players)
#μ̃ = zeros(cc_num_players)

# in 8
pdf_N_tilde = zeros(num_cc, num_games)
pdf_N_lag = zeros(num_cc, num_games)
# in 9
#row_of_i_game_A_cc_game = Vector{Int64}(zeros(cc_num_players))
#row_of_i_game_A_R = Vector{Int64}(zeros(cc_num_players))

# in all: preset A_R memory
A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()



u_sum_A_tilde = zeros(num_cc, n_par)
u_sum_A_R_lag =  zeros(num_cc, n_par)
u_sum_A_lag = zeros(num_cc, n_par)
u_sum_A_R_tilde = zeros(num_cc, n_par)
μ_log_sum_tilde = zeros(num_cc, n_par)
μ_log_sum_lag = zeros(num_cc, n_par)


#tt=3
Random.seed!(10)


## 9.2 full model

# 9.2.1 2:10000
tt=2


@time for tt in 2:10000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

# check convergence pattern
quantile(β_MC[1:5000,3], 0.95)
quantile(β_MC[20001:30000,3], 0.025)
quantile(α_MC[1:5000], 0.95)
quantile(α_MC[20001:30000], 0.005)
plot(σ_ηξ_MC[1:1000])
plot(α_MC[1:30000])
plot(β_MC[1:30000,1])
plot(ϕ_MC[1:30000,1])


## save results
writedlm("τ_η_MC10000.txt", τ_η_MC[1:10000], ',')
writedlm("δ_MC10000.txt", δ_MC[1:10000,:], ',')
writedlm("ω_MC10000.txt", ω_MC[1:10000,:], ',')
writedlm("β_MC10000.txt", β_MC[1:10000,:], ',')
writedlm("α_MC10000.txt", α_MC[1:10000], ',')
writedlm("ρ_MC10000.txt", ρ_MC[1:10000,:], ',')
writedlm("ϕ_MC10000.txt", ϕ_MC[1:10000,:], ',')
writedlm("γ_MC10000.txt", γ_MC[1:10000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[10000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[10000,:,:], ',')
writedlm("σ_ηξ_MC10000.txt", σ_ηξ_MC[1:10000], ',')
writedlm("η_MClast.txt", η_MC[10000,:,:], ',')


# 9.2.2 10001:20000
# read previous iterations
τ_η_MC[1:10000] = readdlm("τ_η_MC10000.txt", ',')
δ_MC[1:10000,:] = readdlm("δ_MC10000.txt", ',')
ω_MC[1:10000,:] = readdlm("ω_MC10000.txt", ',')
β_MC[1:10000,:] = readdlm("β_MC10000.txt", ',')
α_MC[1:10000] = readdlm("α_MC10000.txt", ',')
ρ_MC[1:10000,:] = readdlm("ρ_MC10000.txt", ',')
ϕ_MC[1:10000,:] = readdlm("ϕ_MC10000.txt", ',')
γ_MC[1:10000,:] = readdlm("γ_MC10000.txt", ',')

ν_MC[10000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[10000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[1:10000] = readdlm("σ_ηξ_MC10000.txt", ',')
ξ_MC[10000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[10000,:,:] = readdlm("η_MClast.txt", ',')


# Bayesian MCMC

@time for tt in 10001:20000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC20000.txt", τ_η_MC[10001:20000], ',')
writedlm("δ_MC20000.txt", δ_MC[10001:20000,:], ',')
writedlm("ω_MC20000.txt", ω_MC[10001:20000,:], ',')
writedlm("β_MC20000.txt", β_MC[10001:20000,:], ',')
writedlm("α_MC20000.txt", α_MC[10001:20000], ',')
writedlm("ρ_MC20000.txt", ρ_MC[10001:20000,:], ',')
writedlm("ϕ_MC20000.txt", ϕ_MC[10001:20000,:], ',')
writedlm("γ_MC20000.txt", γ_MC[10001:20000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[20000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[20000,:,:], ',')
writedlm("σ_ηξ_MC20000.txt", σ_ηξ_MC[10001:20000], ',')
writedlm("η_MClast.txt", η_MC[20000,:,:], ',')


# 9.2.3 20001:30000
# read previous iterations
τ_η_MC[10001:20000] = readdlm("τ_η_MC20000.txt", ',')
δ_MC[10001:20000,:] = readdlm("δ_MC20000.txt", ',')
ω_MC[10001:20000,:] = readdlm("ω_MC20000.txt", ',')
β_MC[10001:20000,:] = readdlm("β_MC20000.txt", ',')
α_MC[10001:20000] = readdlm("α_MC20000.txt", ',')
ρ_MC[10001:20000,:] = readdlm("ρ_MC20000.txt", ',')
ϕ_MC[10001:20000,:] = readdlm("ϕ_MC20000.txt", ',')
γ_MC[10001:20000,:] = readdlm("γ_MC20000.txt", ',')

ν_MC[20000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[20000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[10001:20000] = readdlm("σ_ηξ_MC20000.txt", ',')
ξ_MC[20000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[20000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 20001:30000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC30000.txt", τ_η_MC[20001:30000], ',')
writedlm("δ_MC30000.txt", δ_MC[20001:30000,:], ',')
writedlm("ω_MC30000.txt", ω_MC[20001:30000,:], ',')
writedlm("β_MC30000.txt", β_MC[20001:30000,:], ',')
writedlm("α_MC30000.txt", α_MC[20001:30000], ',')
writedlm("ρ_MC30000.txt", ρ_MC[20001:30000,:], ',')
writedlm("ϕ_MC30000.txt", ϕ_MC[20001:30000,:], ',')
writedlm("γ_MC30000.txt", γ_MC[20001:30000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[30000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[30000,:,:], ',')
writedlm("σ_ηξ_MC30000.txt", σ_ηξ_MC[20001:30000], ',')
writedlm("η_MClast.txt", η_MC[30000,:,:], ',')

# 9.2.4 30001:40000
# read previous iterations
τ_η_MC[20001:30000] = readdlm("τ_η_MC30000.txt", ',')
δ_MC[20001:30000,:] = readdlm("δ_MC30000.txt", ',')
ω_MC[20001:30000,:] = readdlm("ω_MC30000.txt", ',')
β_MC[20001:30000,:] = readdlm("β_MC30000.txt", ',')
α_MC[20001:30000] = readdlm("α_MC30000.txt", ',')
ρ_MC[20001:30000,:] = readdlm("ρ_MC30000.txt", ',')
ϕ_MC[20001:30000,:] = readdlm("ϕ_MC30000.txt", ',')
γ_MC[20001:30000,:] = readdlm("γ_MC30000.txt", ',')

ν_MC[30000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[30000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[20001:30000] = readdlm("σ_ηξ_MC30000.txt", ',')
ξ_MC[30000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[30000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 30001:40000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC40000.txt", τ_η_MC[30001:40000], ',')
writedlm("δ_MC40000.txt", δ_MC[30001:40000,:], ',')
writedlm("ω_MC40000.txt", ω_MC[30001:40000,:], ',')
writedlm("β_MC40000.txt", β_MC[30001:40000,:], ',')
writedlm("α_MC40000.txt", α_MC[30001:40000], ',')
writedlm("ρ_MC40000.txt", ρ_MC[30001:40000,:], ',')
writedlm("ϕ_MC40000.txt", ϕ_MC[30001:40000,:], ',')
writedlm("γ_MC40000.txt", γ_MC[30001:40000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[40000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[40000,:,:], ',')
writedlm("σ_ηξ_MC40000.txt", σ_ηξ_MC[30001:40000], ',')
writedlm("η_MClast.txt", η_MC[40000,:,:], ',')


# 9.2.5 40001:50000
# read previous iterations
τ_η_MC[30001:40000] = readdlm("τ_η_MC40000.txt", ',')
δ_MC[30001:40000,:] = readdlm("δ_MC40000.txt", ',')
ω_MC[30001:40000,:] = readdlm("ω_MC40000.txt", ',')
β_MC[30001:40000,:] = readdlm("β_MC40000.txt", ',')
α_MC[30001:40000] = readdlm("α_MC40000.txt", ',')
ρ_MC[30001:40000,:] = readdlm("ρ_MC40000.txt", ',')
ϕ_MC[30001:40000,:] = readdlm("ϕ_MC40000.txt", ',')
γ_MC[30001:40000,:] = readdlm("γ_MC40000.txt", ',')

ν_MC[40000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[40000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[30001:40000] = readdlm("σ_ηξ_MC40000.txt", ',')
ξ_MC[40000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[40000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 40001:50000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC50000.txt", τ_η_MC[40001:50000], ',')
writedlm("δ_MC50000.txt", δ_MC[40001:50000,:], ',')
writedlm("ω_MC50000.txt", ω_MC[40001:50000,:], ',')
writedlm("β_MC50000.txt", β_MC[40001:50000,:], ',')
writedlm("α_MC50000.txt", α_MC[40001:50000], ',')
writedlm("ρ_MC50000.txt", ρ_MC[40001:50000,:], ',')
writedlm("ϕ_MC50000.txt", ϕ_MC[40001:50000,:], ',')
writedlm("γ_MC50000.txt", γ_MC[40001:50000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[50000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[50000,:,:], ',')
writedlm("σ_ηξ_MC50000.txt", σ_ηξ_MC[40001:50000], ',')
writedlm("η_MClast.txt", η_MC[50000,:,:], ',')

# 9.2.6 50001:60000
# read previous iterations
τ_η_MC[40001:50000] = readdlm("τ_η_MC50000.txt", ',')
δ_MC[40001:50000,:] = readdlm("δ_MC50000.txt", ',')
ω_MC[40001:50000,:] = readdlm("ω_MC50000.txt", ',')
β_MC[40001:50000,:] = readdlm("β_MC50000.txt", ',')
α_MC[40001:50000] = readdlm("α_MC50000.txt", ',')
ρ_MC[40001:50000,:] = readdlm("ρ_MC50000.txt", ',')
ϕ_MC[40001:50000,:] = readdlm("ϕ_MC50000.txt", ',')
γ_MC[40001:50000,:] = readdlm("γ_MC50000.txt", ',')

ν_MC[50000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[50000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[40001:50000] = readdlm("σ_ηξ_MC50000.txt", ',')
ξ_MC[50000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[50000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 50001:60000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC60000.txt", τ_η_MC[50001:60000], ',')
writedlm("δ_MC60000.txt", δ_MC[50001:60000,:], ',')
writedlm("ω_MC60000.txt", ω_MC[50001:60000,:], ',')
writedlm("β_MC60000.txt", β_MC[50001:60000,:], ',')
writedlm("α_MC60000.txt", α_MC[50001:60000], ',')
writedlm("ρ_MC60000.txt", ρ_MC[50001:60000,:], ',')
writedlm("ϕ_MC60000.txt", ϕ_MC[50001:60000,:], ',')
writedlm("γ_MC60000.txt", γ_MC[50001:60000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[60000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[60000,:,:], ',')
writedlm("σ_ηξ_MC60000.txt", σ_ηξ_MC[50001:60000], ',')
writedlm("η_MClast.txt", η_MC[60000,:,:], ',')


# 9.2.7 60001:70000
# read previous iterations
τ_η_MC[50001:60000] = readdlm("τ_η_MC60000.txt", ',')
δ_MC[50001:60000,:] = readdlm("δ_MC60000.txt", ',')
ω_MC[50001:60000,:] = readdlm("ω_MC60000.txt", ',')
β_MC[50001:60000,:] = readdlm("β_MC60000.txt", ',')
α_MC[50001:60000] = readdlm("α_MC60000.txt", ',')
ρ_MC[50001:60000,:] = readdlm("ρ_MC60000.txt", ',')
ϕ_MC[50001:60000,:] = readdlm("ϕ_MC60000.txt", ',')
γ_MC[50001:60000,:] = readdlm("γ_MC60000.txt", ',')

ν_MC[60000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[60000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[50001:60000] = readdlm("σ_ηξ_MC60000.txt", ',')
ξ_MC[60000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[60000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 60001:70000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC70000.txt", τ_η_MC[60001:70000], ',')
writedlm("δ_MC70000.txt", δ_MC[60001:70000,:], ',')
writedlm("ω_MC70000.txt", ω_MC[60001:70000,:], ',')
writedlm("β_MC70000.txt", β_MC[60001:70000,:], ',')
writedlm("α_MC70000.txt", α_MC[60001:70000], ',')
writedlm("ρ_MC70000.txt", ρ_MC[60001:70000,:], ',')
writedlm("ϕ_MC70000.txt", ϕ_MC[60001:70000,:], ',')
writedlm("γ_MC70000.txt", γ_MC[60001:70000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[70000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[70000,:,:], ',')
writedlm("σ_ηξ_MC70000.txt", σ_ηξ_MC[60001:70000], ',')
writedlm("η_MClast.txt", η_MC[70000,:,:], ',')

# 9.2.8 70001:80000
# read previous iterations
τ_η_MC[60001:70000] = readdlm("τ_η_MC70000.txt", ',')
δ_MC[60001:70000,:] = readdlm("δ_MC70000.txt", ',')
ω_MC[60001:70000,:] = readdlm("ω_MC70000.txt", ',')
β_MC[60001:70000,:] = readdlm("β_MC70000.txt", ',')
α_MC[60001:70000] = readdlm("α_MC70000.txt", ',')
ρ_MC[60001:70000,:] = readdlm("ρ_MC70000.txt", ',')
ϕ_MC[60001:70000,:] = readdlm("ϕ_MC70000.txt", ',')
γ_MC[60001:70000,:] = readdlm("γ_MC70000.txt", ',')

ν_MC[70000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[70000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[60001:70000] = readdlm("σ_ηξ_MC70000.txt", ',')
ξ_MC[70000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[70000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 70001:80000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC80000.txt", τ_η_MC[70001:80000], ',')
writedlm("δ_MC80000.txt", δ_MC[70001:80000,:], ',')
writedlm("ω_MC80000.txt", ω_MC[70001:80000,:], ',')
writedlm("β_MC80000.txt", β_MC[70001:80000,:], ',')
writedlm("α_MC80000.txt", α_MC[70001:80000], ',')
writedlm("ρ_MC80000.txt", ρ_MC[70001:80000,:], ',')
writedlm("ϕ_MC80000.txt", ϕ_MC[70001:80000,:], ',')
writedlm("γ_MC80000.txt", γ_MC[70001:80000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[80000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[80000,:,:], ',')
writedlm("σ_ηξ_MC80000.txt", σ_ηξ_MC[70001:80000], ',')
writedlm("η_MClast.txt", η_MC[80000,:,:], ',')

# 9.2.9 80001:90000
# read previous iterations
τ_η_MC[70001:80000] = readdlm("τ_η_MC80000.txt", ',')
δ_MC[70001:80000,:] = readdlm("δ_MC80000.txt", ',')
ω_MC[70001:80000,:] = readdlm("ω_MC80000.txt", ',')
β_MC[70001:80000,:] = readdlm("β_MC80000.txt", ',')
α_MC[70001:80000] = readdlm("α_MC80000.txt", ',')
ρ_MC[70001:80000,:] = readdlm("ρ_MC80000.txt", ',')
ϕ_MC[70001:80000,:] = readdlm("ϕ_MC80000.txt", ',')
γ_MC[70001:80000,:] = readdlm("γ_MC80000.txt", ',')

ν_MC[80000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[80000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[70001:80000] = readdlm("σ_ηξ_MC80000.txt", ',')
ξ_MC[80000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[80000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 80001:90000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC90000.txt", τ_η_MC[80001:90000], ',')
writedlm("δ_MC90000.txt", δ_MC[80001:90000,:], ',')
writedlm("ω_MC90000.txt", ω_MC[80001:90000,:], ',')
writedlm("β_MC90000.txt", β_MC[80001:90000,:], ',')
writedlm("α_MC90000.txt", α_MC[80001:90000], ',')
writedlm("ρ_MC90000.txt", ρ_MC[80001:90000,:], ',')
writedlm("ϕ_MC90000.txt", ϕ_MC[80001:90000,:], ',')
writedlm("γ_MC90000.txt", γ_MC[80001:90000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[90000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[90000,:,:], ',')
writedlm("σ_ηξ_MC90000.txt", σ_ηξ_MC[80001:90000], ',')
writedlm("η_MClast.txt", η_MC[90000,:,:], ',')

# 9.2.9 90001:100000
# read previous iterations
τ_η_MC[90001:100000] = readdlm("τ_η_MC90000.txt", ',')
δ_MC[90001:100000,:] = readdlm("δ_MC90000.txt", ',')
ω_MC[90001:100000,:] = readdlm("ω_MC90000.txt", ',')
β_MC[90001:100000,:] = readdlm("β_MC90000.txt", ',')
α_MC[90001:100000] = readdlm("α_MC90000.txt", ',')
ρ_MC[90001:100000,:] = readdlm("ρ_MC90000.txt", ',')
ϕ_MC[90001:100000,:] = readdlm("ϕ_MC90000.txt", ',')
γ_MC[90001:100000,:] = readdlm("γ_MC90000.txt", ',')

ν_MC[90000] = Dict{String, Matrix{Float64}}()
for cc in 1:num_cc
    ν_MC[90000][Country_List[cc]] = readdlm("$(cc)_ν_MClast.txt", ',')
end

σ_ηξ_MC[80001:90000] = readdlm("σ_ηξ_MC90000.txt", ',')
ξ_MC[90000,:,:] = readdlm("ξ_MClast.txt", ',')
η_MC[90000,:,:] = readdlm("η_MClast.txt", ',')

# Bayesian MCMC
@time for tt in 90001:100000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = ϕ_MC[tt-1,:]
    γ_MC_lag = γ_MC[tt-1,:]
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    #η_MC_tt = reshape(price, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ̃ = ϕ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ϕ, proposal_var_ϕ)) # multiply μ_ϕ, make less variation
    
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt,  ν_MC_lag, ϕ̃, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end
    
      # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ̃, ξ_MC_lag)
        end
    end    
     # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ϕ, Σ_ϕ), ϕ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ϕ_MC_tt = ϕ̃
    else
        ϕ_MC_tt = copy(ϕ_MC_lag)
    end

    ## 6
    println("tt=$tt, step 6")
    ##(a)
    γ̃ = γ_MC_lag + rand(Distributions.MvNormal(proposal_mean_γ, proposal_var_γ)) # multiply μ_γ, make less variation
    ##(b)
    ## for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_γ, Σ_γ), γ_MC_lag)

    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            np += 1
            μ_log_sum_tilde[cc,np] =  gen_μ_log_sum(cc, key, m,  γ̃,ν_MC_lag)
            μ_log_sum_lag[cc,np] =  gen_μ_log_sum(cc, key, m, γ_MC_lag,ν_MC_lag)
        end
    end
   
    a_bar = (
            log(pdf_tilde) + sum(μ_log_sum_tilde)
            -
            (log(pdf_lag) + sum(μ_log_sum_lag))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        γ_MC_tt = γ̃
    else
        γ_MC_tt = copy(γ_MC_lag)
    end
    ## 7
    
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = deepcopy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = deepcopy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = deepcopy(ν_MC_lag)


    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, - τ_η_MC_tt^(-1/2),  τ_η_MC_tt^(-1/2))) # control  σ̃_ηξ so that it is in the defined range 

    #σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, max(- τ_η_MC_tt^(-1/2), - τ_η_MC_lag^(-1/2)),  min(τ_η_MC_tt^(-1/2), τ_η_MC_lag^(-1/2)))) # control  σ̃_ηξ so that it is in the defined range 
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end
    #σ_ηξ_MC_tt = 1.0
    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c) 
                #pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                #pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                pdf_tilde = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt))), ξ̃_bb) 
                pdf_lag = pdf(Normal(σ_ηξ_MC_tt * τ_η_MC_tt * η_MC_tt[cc,bb],sqrt(abs(1- σ_ηξ_MC_tt^2 * τ_η_MC_tt) )), ξ_temp_bb) 

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end

## save results
writedlm("τ_η_MC100000.txt", τ_η_MC[90001:100000], ',')
writedlm("δ_MC100000.txt", δ_MC[90001:100000,:], ',')
writedlm("ω_MC100000.txt", ω_MC[90001:100000,:], ',')
writedlm("β_MC100000.txt", β_MC[90001:100000,:], ',')
writedlm("α_MC100000.txt", α_MC[90001:100000], ',')
writedlm("ρ_MC100000.txt", ρ_MC[90001:100000,:], ',')
writedlm("ϕ_MC100000.txt", ϕ_MC[90001:100000,:], ',')
writedlm("γ_MC100000.txt", γ_MC[90001:100000,:], ',')

for cc in 1:num_cc
    writedlm("$(cc)_ν_MClast.txt", ν_MC[100000][Country_List[cc]], ',')
end

writedlm("ξ_MClast.txt", ξ_MC[100000,:,:], ',')
writedlm("σ_ηξ_MC100000.txt", σ_ηξ_MC[90001:100000], ',')
writedlm("η_MClast.txt", η_MC[100000,:,:], ',')


# temp
# 9.3 read results

τ_η_MC[1:5000] = readdlm("τ_η_MC5000.txt", ',')
δ_MC[1:5000,:] = readdlm("δ_MC5000.txt", ',')
ω_MC[1:5000,:] = readdlm("ω_MC5000.txt", ',')
β_MC[1:5000,:] = readdlm("β_MC5000.txt", ',')
α_MC[1:5000] = readdlm("α_MC5000.txt", ',')
ρ_MC[1:5000,:] = readdlm("ρ_MC5000.txt", ',')
ϕ_MC[1:5000,:] = readdlm("ϕ_MC5000.txt", ',')
γ_MC[1:5000,:] = readdlm("γ_MC5000.txt", ',')
σ_ηξ_MC[1:5000] = readdlm("σ_ηξ_MC5000.txt", ',')



τ_η_MC[5000:10000] = readdlm("τ_η_MC5000_10000.txt", ',')
δ_MC[5000:10000,:] = readdlm("δ_MC5000_10000.txt", ',')
ω_MC[5000:10000,:] = readdlm("ω_MC5000_10000.txt", ',')
β_MC[5000:10000,:] = readdlm("β_MC5000_10000.txt", ',')
α_MC[5000:10000] = readdlm("α_MC5000_10000.txt", ',')
ρ_MC[5000:10000,:] = readdlm("ρ_MC5000_10000.txt", ',')
ϕ_MC[5000:10000,:] = readdlm("ϕ_MC5000_10000.txt", ',')
γ_MC[5000:10000,:] = readdlm("γ_MC5000_10000.txt", ',')
σ_ηξ_MC[5000:10000] = readdlm("σ_ηξ_MC5000_10000.txt", ',')


plot(σ_ηξ_MC[5000:10000])

# price

plot(α_MC[1:10000])
price_full = zeros(8)
price_full[1] = mean(α_MC[5001:10000])
price_full[2] = std(α_MC[5001:10000])
price_full[3] = quantile(α_MC[5001:10000], 0.95)
price_full[4] = quantile(α_MC[5001:10000], 0.05)
price_full[5] = quantile(α_MC[5001:10000], 0.975)
price_full[6] = quantile(α_MC[5001:10000], 0.025)
price_full[7] = quantile(α_MC[5001:10000], 0.995)
price_full[8] = quantile(α_MC[5001:10000], 0.005)

writedlm("price_full.txt", price_full, ',')
# Rating
plot(β_MC[5001:10000,3])
rating_full = zeros(8)
rating_full[1] = mean(β_MC[5001:10000,3])
rating_full[2] = std(β_MC[5001:10000,3])
rating_full[3] = quantile(β_MC[5001:10000,3], 0.95)
rating_full[4] = quantile(β_MC[5001:10000,3], 0.05)
rating_full[5] = quantile(β_MC[5001:10000,3], 0.975)
rating_full[6] = quantile(β_MC[5001:10000,3], 0.025)
rating_full[7] = quantile(β_MC[5001:10000,3], 0.995)
rating_full[8] = quantile(β_MC[5001:10000,3], 0.005)
writedlm("rating_full.txt", price_full, ',')

# Multiplayer
plot(β_MC[5001:10000,2])
is_mul_full = zeros(8)
is_mul_full[1] = mean(β_MC[5001:10000,2])
is_mul_full[2] = std(β_MC[5001:10000,2])
is_mul_full[3] = quantile(β_MC[5001:10000,2], 0.95)
is_mul_full[4] = quantile(β_MC[5001:10000,2], 0.05)
is_mul_full[5] = quantile(β_MC[5001:10000,2], 0.975)
is_mul_full[6] = quantile(β_MC[5001:10000,2], 0.025)
is_mul_full[7] = quantile(β_MC[5001:10000,2], 0.995)
is_mul_full[8] = quantile(β_MC[5001:10000,2], 0.005)
writedlm("is_mul_full.txt", price_full, ',')
# Age
plot(β_MC[5001:10000,4])
age_full = zeros(8)
age_full[1] = mean(β_MC[5001:10000,4])
age_full[2] = std(β_MC[5001:10000,4])
age_full[3] = quantile(β_MC[5001:10000,4], 0.95)
age_full[4] = quantile(β_MC[5001:10000,4], 0.05)
age_full[5] = quantile(β_MC[5001:10000,4], 0.975)
age_full[6] = quantile(β_MC[5001:10000,4], 0.025)
age_full[7] = quantile(β_MC[5001:10000,4], 0.995)
age_full[8] = quantile(β_MC[5001:10000,4], 0.005)
writedlm("age_full.txt", age_full, ',')

# network single
plot(ϕ_MC[5001:10000,1])
net_sin_full = zeros(8)
net_sin_full[1] = mean(ϕ_MC[5001:10000,1])
net_sin_full[2] = std(ϕ_MC[5001:10000,1])
net_sin_full[3] = quantile(ϕ_MC[5001:10000,1], 0.95)
net_sin_full[4] = quantile(ϕ_MC[5001:10000,1], 0.05)
net_sin_full[5] = quantile(ϕ_MC[5001:10000,1], 0.975)
net_sin_full[6] = quantile(ϕ_MC[5001:10000,1], 0.025)
net_sin_full[7] = quantile(ϕ_MC[5001:10000,1], 0.995)
net_sin_full[8] = quantile(ϕ_MC[5001:10000,1], 0.005)
writedlm("net_sin_full.txt", age_full, ',')


# network multi
plot(ϕ_MC[5001:10000,2])
net_mul_full = zeros(8)
net_mul_full[1] = mean(ϕ_MC[5001:10000,2])
net_mul_full[2] = std(ϕ_MC[5001:10000,2])
net_mul_full[3] = quantile(ϕ_MC[5001:10000,2], 0.95)
net_mul_full[4] = quantile(ϕ_MC[5001:10000,2], 0.05)
net_mul_full[5] = quantile(ϕ_MC[5001:10000,2], 0.975)
net_mul_full[6] = quantile(ϕ_MC[5001:10000,2], 0.025)
net_mul_full[7] = quantile(ϕ_MC[5001:10000,2], 0.995)
net_mul_full[8] = quantile(ϕ_MC[5001:10000,2], 0.005)
writedlm("net_mul_full.txt", age_full, ',')

β_est = mean(β_MC[8001:10000,:], dims = 1)
α_est = mean(α_MC[8001:10000,:])

ρ_est = mean(ρ_MC[8001:10000,:], dims = 1)
ϕ_est = mean(ϕ_MC[8001:10000,:], dims = 1)




R=10000
game_benchmark = 1
rate = 0.05


#findmax(Dict(k => length(v) for (k,v) in partition[Country_List[1]]))[2]


A_R_model_fit = Dict{Int64, Matrix{Float64}}() # first Int64: country, Int64: partition key
A_R_model_obs = Dict{Int64, Matrix{Float64}}() 
#A_cc_game_density, A_cc_game_dispersion = MC_A_density(R,cc, key, m, β, α, ρ, ν, ϕ, ξ,game_benchmark, rate) 

#A_cc_game_density, A_cc_game_dispersion = MC_A_density(R,cc, key, m, β, α, ρ, ν, ϕ, ξ,game_benchmark, rate) 


ρ_est
ρ_est = zeros(1,4)

@time for cc in 1:num_cc
    A_R_model_fit[cc] = zeros(50,1)
    A_R_model_obs[cc] = zeros(50,1)
    #key = findmax(Dict(k => length(v) for (k,v) in partition[Country_List[cc]]))[2]
    for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
        println("country $cc, key $key") 
        A_R_model_fit[cc] += 1/n_par * mean(MC_A_density(R,cc, key, m, β_est, α_est, ρ_est,  ν, ϕ_est, ξ, game_benchmark, rate)[1][5000:10000,:], dims = 1)' # * length(m.partition_index[Country_List[cc]][key]) 
        A_R_model_obs[cc] += 1/n_par *  mean(m.game_play[Country_List[cc]][:, m.partition_index[Country_List[cc]][key]], dims =2)
    end
end


total_player_sample = sum(sum(length(m.partition_index[Country_List[cc]][key])  for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]) for cc in 1:num_cc)

## 9.4.1 table for model fit
Country_List
# games: 107410, 214950, 236430, 222880, 209170, 206420


market_share_fit = zeros(7)
market_share_obs = zeros(7)
# 107410

market_share_fit[1] = ((A_R_model_fit[1][1]
                        + A_R_model_fit[2][1]
                        + A_R_model_fit[3][1]
                        + A_R_model_fit[4][2]
                        + A_R_model_fit[5][2]
                        + A_R_model_fit[6][1]
                        + A_R_model_fit[7][1]
                        + A_R_model_fit[8][1]
                        + A_R_model_fit[9][1]
                        + A_R_model_fit[10][1]
                        )/10#/total_player_sample #/ sum(values(m.num_players))
                        )


market_share_obs[1] = ((A_R_model_obs[1][1]
                        + A_R_model_obs[2][1]
                        + A_R_model_obs[3][1]
                        + A_R_model_obs[4][2]
                        + A_R_model_obs[5][2]
                        + A_R_model_obs[6][1]
                        + A_R_model_obs[7][1]
                        + A_R_model_obs[8][1]
                        + A_R_model_obs[9][1]
                        + A_R_model_obs[10][1]
                        )/10#/total_player_sample #/ sum(values(m.num_players))
                        )       
                        
                        
              

#=
market_share_sample[1] =  ((
                            sum(m.game_play[Country_List[1]][1,:])
                        +   sum(m.game_play[Country_List[2]][1,:])
                        +   sum(m.game_play[Country_List[3]][1,:])
                        +   sum(m.game_play[Country_List[4]][2,:])
                        +   sum(m.game_play[Country_List[5]][2,:])
                        +   sum(m.game_play[Country_List[6]][1,:])
                        +   sum(m.game_play[Country_List[7]][1,:])
                        +   sum(m.game_play[Country_List[8]][1,:])
                        +   sum(m.game_play[Country_List[9]][1,:])
                        +   sum(m.game_play[Country_List[10]][1,:])
                        ) / sum(values(m.num_players))
                        )    
=#                        
                        
# 214950
market_share_fit[2] = ((A_R_model_fit[1][3]
                        + A_R_model_fit[2][2]
                        + A_R_model_fit[3][2]
                        + A_R_model_fit[4][9]
                        + A_R_model_fit[5][4]
                        + A_R_model_fit[6][3]
                        + A_R_model_fit[7][2]
                        + A_R_model_fit[8][2]
                        + A_R_model_fit[9][2]
                        + A_R_model_fit[10][3]
                        ) /10#/total_player_sample#/ sum(values(m.num_players))
                        )




market_share_obs[2] = ((A_R_model_obs[1][3]
                        + A_R_model_obs[2][2]
                        + A_R_model_obs[3][2]
                        + A_R_model_obs[4][9]
                        + A_R_model_obs[5][4]
                        + A_R_model_obs[6][3]
                        + A_R_model_obs[7][2]
                        + A_R_model_obs[8][2]
                        + A_R_model_obs[9][2]
                        + A_R_model_obs[10][3]
                        ) /10#/total_player_sample#/ sum(values(m.num_players))
                        )








#=
market_share_sample[2] =  ((
                            sum(m.game_play[Country_List[1]][3,:])
                        +   sum(m.game_play[Country_List[2]][2,:])
                        +   sum(m.game_play[Country_List[3]][2,:])
                        +   sum(m.game_play[Country_List[4]][9,:])
                        +   sum(m.game_play[Country_List[5]][4,:])
                        +   sum(m.game_play[Country_List[6]][3,:])
                        +   sum(m.game_play[Country_List[7]][2,:])
                        +   sum(m.game_play[Country_List[8]][2,:])
                        +   sum(m.game_play[Country_List[9]][2,:])
                        +   sum(m.game_play[Country_List[10]][3,:])
                        ) / sum(values(m.num_players))
                        )     
=#

#236430
market_share_fit[3] = ((A_R_model_fit[1][4]
                        + A_R_model_fit[2][3]
                        + A_R_model_fit[3][4]
                        + A_R_model_fit[4][1]
                        + A_R_model_fit[5][1]
                        + A_R_model_fit[6][2]
                        + A_R_model_fit[7][3]
                        + A_R_model_fit[8][4]
                        + A_R_model_fit[9][4]
                        + A_R_model_fit[10][2]
                        ) /10#/total_player_sample#/ sum(values(m.num_players))
                        )




market_share_obs[3] = ((A_R_model_obs[1][4]
                        + A_R_model_obs[2][3]
                        + A_R_model_obs[3][4]
                        + A_R_model_obs[4][1]
                        + A_R_model_obs[5][1]
                        + A_R_model_obs[6][2]
                        + A_R_model_obs[7][3]
                        + A_R_model_obs[8][4]
                        + A_R_model_obs[9][4]
                        + A_R_model_obs[10][2]
                        ) /10#/total_player_sample#/ sum(values(m.num_players))
                        )






#=
market_share_sample[3] =  ((
                            sum(m.game_play[Country_List[1]][4,:])
                        +   sum(m.game_play[Country_List[2]][3,:])
                        +   sum(m.game_play[Country_List[3]][4,:])
                        +   sum(m.game_play[Country_List[4]][1,:])
                        +   sum(m.game_play[Country_List[5]][1,:])
                        +   sum(m.game_play[Country_List[6]][2,:])
                        +   sum(m.game_play[Country_List[7]][3,:])
                        +   sum(m.game_play[Country_List[8]][4,:])
                        +   sum(m.game_play[Country_List[9]][4,:])
                        +   sum(m.game_play[Country_List[10]][2,:])
                        ) / sum(values(m.num_players))
                        )     
=#


#222880
market_share_fit[4] = ((A_R_model_fit[1][7]
+ A_R_model_fit[2][4]
+ A_R_model_fit[3][5]
+ A_R_model_fit[4][3]
+ A_R_model_fit[5][3]
+ A_R_model_fit[6][4]
+ A_R_model_fit[7][5]
+ A_R_model_fit[8][5]
+ A_R_model_fit[9][5]
+ A_R_model_fit[10][5]
) /10#/total_player_sample#/ sum(values(m.num_players))
)




market_share_obs[4] = ((A_R_model_obs[1][7]
+ A_R_model_obs[2][4]
+ A_R_model_obs[3][5]
+ A_R_model_obs[4][3]
+ A_R_model_obs[5][3]
+ A_R_model_obs[6][4]
+ A_R_model_obs[7][5]
+ A_R_model_obs[8][5]
+ A_R_model_obs[9][5]
+ A_R_model_obs[10][5]
) /10#/total_player_sample#/ sum(values(m.num_players))
)





#=
market_share_sample[4] =  ((
    sum(m.game_play[Country_List[1]][7,:])
+   sum(m.game_play[Country_List[2]][4,:])
+   sum(m.game_play[Country_List[3]][5,:])
+   sum(m.game_play[Country_List[4]][3,:])
+   sum(m.game_play[Country_List[5]][3,:])
+   sum(m.game_play[Country_List[6]][4,:])
+   sum(m.game_play[Country_List[7]][5,:])
+   sum(m.game_play[Country_List[8]][5,:])
+   sum(m.game_play[Country_List[9]][5,:])
+   sum(m.game_play[Country_List[10]][5,:])
) / sum(values(m.num_players))
)     
=#


#206420
market_share_fit[6] = ((A_R_model_fit[1][6]
+ A_R_model_fit[2][7]
+ A_R_model_fit[3][6]
+ A_R_model_fit[4][10]
+ A_R_model_fit[5][5]
+ A_R_model_fit[6][6]
+ A_R_model_fit[7][7]
+ A_R_model_fit[8][6]
+ A_R_model_fit[9][7]
+ A_R_model_fit[10][7]
) /10#/total_player_sample#/ sum(values(m.num_players))
)

market_share_obs[6] = ((A_R_model_obs[1][6]
+ A_R_model_obs[2][7]
+ A_R_model_obs[3][6]
+ A_R_model_obs[4][10]
+ A_R_model_obs[5][5]
+ A_R_model_obs[6][6]
+ A_R_model_obs[7][7]
+ A_R_model_obs[8][6]
+ A_R_model_obs[9][7]
+ A_R_model_obs[10][7]
) /10#/total_player_sample#/ sum(values(m.num_players))
)



#=

market_share_sample[5] =  ((
    sum(m.game_play[Country_List[1]][2,:])
+   sum(m.game_play[Country_List[2]][5,:])
+   sum(m.game_play[Country_List[3]][3,:])
+   sum(m.game_play[Country_List[4]][5,:])
+   sum(m.game_play[Country_List[5]][9,:])
+   sum(m.game_play[Country_List[6]][5,:])
+   sum(m.game_play[Country_List[7]][4,:])
+   sum(m.game_play[Country_List[8]][3,:])
+   sum(m.game_play[Country_List[9]][3,:])
+   sum(m.game_play[Country_List[10]][4,:])
) / sum(values(m.num_players))
)     



=#


#209160
market_share_fit[5] = ((A_R_model_fit[1][5]
+ A_R_model_fit[2][19]
+ A_R_model_fit[3][7]
+ A_R_model_fit[4][21]
+ A_R_model_fit[5][29]
+ A_R_model_fit[6][17]
+ A_R_model_fit[7][13]
+ A_R_model_fit[8][10]
+ A_R_model_fit[9][6]
+ A_R_model_fit[10][13]
) /10#/total_player_sample#/ sum(values(m.num_players))
)

market_share_obs[5] = ((A_R_model_obs[1][5]
+ A_R_model_obs[2][19]
+ A_R_model_obs[3][7]
+ A_R_model_obs[4][21]
+ A_R_model_obs[5][29]
+ A_R_model_obs[6][17]
+ A_R_model_obs[7][13]
+ A_R_model_obs[8][10]
+ A_R_model_obs[9][6]
+ A_R_model_obs[10][13]
) /10#/total_player_sample#/ sum(values(m.num_players))
)



#209170
market_share_fit[7] = ((A_R_model_fit[1][2]
+ A_R_model_fit[2][5]
+ A_R_model_fit[3][3]
+ A_R_model_fit[4][5]
+ A_R_model_fit[5][9]
+ A_R_model_fit[6][5]
+ A_R_model_fit[7][4]
+ A_R_model_fit[8][3]
+ A_R_model_fit[9][3]
+ A_R_model_fit[10][4]
) /10#/total_player_sample#/ sum(values(m.num_players))
)
#market_share_fit[7] = (A_R_model_fit[1][2] + A_R_model_fit[1][5])/2

market_share_obs[7] = ((A_R_model_obs[1][2]
+ A_R_model_obs[2][5]
+ A_R_model_obs[3][3]
+ A_R_model_obs[4][5]
+ A_R_model_obs[5][9]
+ A_R_model_obs[6][5]
+ A_R_model_obs[7][4]
+ A_R_model_obs[8][3]
+ A_R_model_obs[9][3]
+ A_R_model_obs[10][4]
) /10#/total_player_sample#/ sum(values(m.num_players))
)

## HHI
sum(market_share_fit.^2) + (0.5-sum(market_share_fit))^2 + 0.25
sum(market_share_obs.^2) + (0.5-sum(market_share_obs))^2 + 0.25

std(market_share_fit)
std(market_share_obs)

m.game_play
total_players
sum(values(m.num_players))
print(m.num_players)




## 9.5 countefactual: varying Arma 3 price (appid 107410)


R=10000
game_benchmark = 1
rate = 0.05

## p = log(9.99)
for cc in 1:num_cc
    m.top_game_info[Country_List[cc]][m.top_game_info[Country_List[cc]].appid .== 107410, :Price] = [log(69.99)]
end

# with net
@time for cc in 1:num_cc
    A_R_model_fit[cc] = zeros(50,1)
    #key = findmax(Dict(k => length(v) for (k,v) in partition[Country_List[cc]]))[2]
    for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
        println("country $cc, key $key") 
        A_R_model_fit[cc] += 1/n_par * mean(MC_A_density(R,cc, key, m, β_est, α_est, ρ_est,  ν, ϕ_est, ξ, game_benchmark, rate)[1][5000:10000,:], dims = 1)' # * length(m.partition_index[Country_List[cc]][key]) 
    end
end


market_share_counter = ((A_R_model_fit[1][1]
                        + A_R_model_fit[2][1]
                        + A_R_model_fit[3][1]
                        + A_R_model_fit[4][2]
                        + A_R_model_fit[5][2]
                        + A_R_model_fit[6][1]
                        + A_R_model_fit[7][1]
                        + A_R_model_fit[8][1]
                        + A_R_model_fit[9][1]
                        + A_R_model_fit[10][1]
                        )/10#/total_player_sample #/ sum(values(m.num_players))
                        )

                        sum(values(A_num_players))

# no net


R=10000
game_benchmark = 1
rate = 0.05

## p = log(9.99)
for cc in 1:num_cc
    m.top_game_info[Country_List[cc]][m.top_game_info[Country_List[cc]].appid .== 107410, :Price] = [log(69.99)]
end

ϕ_est_temp = zeros(2)
@time for cc in 1:num_cc
    A_R_model_fit[cc] = zeros(50,1)
    #key = findmax(Dict(k => length(v) for (k,v) in partition[Country_List[cc]]))[2]
    for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
        println("country $cc, key $key") 
        A_R_model_fit[cc] += 1/n_par * mean(MC_A_density(R,cc, key, m, β_est, α_est, ρ_est,  ν, ϕ_est_temp, ξ, game_benchmark, rate)[1][5000:10000,:], dims = 1)' # * length(m.partition_index[Country_List[cc]][key]) 
    end
end


market_share_fit[1] = ((A_R_model_fit[1][1]
                        + A_R_model_fit[2][1]
                        + A_R_model_fit[3][1]
                        + A_R_model_fit[4][2]
                        + A_R_model_fit[5][2]
                        + A_R_model_fit[6][1]
                        + A_R_model_fit[7][1]
                        + A_R_model_fit[8][1]
                        + A_R_model_fit[9][1]
                        + A_R_model_fit[10][1]
                        )/10#/total_player_sample #/ sum(values(m.num_players))
                        )


## 9.3 different version for ϕ = 0, matching BLP

@time for tt in 2:10000
    τ_η_MC_lag = τ_η_MC[tt-1] 
    δ_MC_lag = δ_MC[tt-1,:] 
    ω_MC_lag = ω_MC[tt-1,:] 
    β_MC_lag = β_MC[tt-1,:]
    α_MC_lag = α_MC[tt-1]
    ρ_MC_lag = ρ_MC[tt-1,:]
    ϕ_MC_lag = zeros(2)
    γ_MC_lag = zeros(num_rc+2) 
    ν_MC_lag = ν_MC[tt-1]
    ξ_MC_lag = ξ_MC[tt-1,:,:]
    σ_ηξ_MC_lag = σ_ηξ_MC[tt-1]
    η_MC_lag = η_MC[tt-1,:,:]

    println("tt=$tt,")
    ## 1
    println("tt=$tt, step 1")
    α_τ_η_post = α_τ_η + num_games * num_cc/2 + n_z/2
    #
    τ_η_MC_tt = (rand(Gamma(α_τ_η_post,
                            (
                                β_τ_η + sum(η_MC_lag .^2)/2 + (δ_MC_lag - μ_δ)' * inv(Σ_δ) * (δ_MC_lag - μ_δ)/2
                            )^(-1)
                        ))
                )
    ## 2
    println("tt=$tt, step 2")
    δ_MC_tt = (rand(Distributions.MvNormal(inv_Q_δ_l_δ, Matrix(Hermitian(τ_η_MC_tt^(-1) * inv_Q_δ)))
                      )
                )
    η_MC_tt = reshape(price - IV * δ_MC_tt, (num_games,num_cc))'
    ## 3
    println("tt=$tt, step 3")
    # (a) propose tilde from random walk
    ω̃ = ω_MC_lag + rand(Distributions.MvNormal(proposal_mean_ω, proposal_var_ω)) # multiply μ_ω, make less variation
    β̃ = ω̃[1:k]
    α̃ = ω̃[k+1]
    
    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β̃, α̃, ρ_MC_lag,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end

   
    # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_lag, α_MC_lag, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β̃, α̃, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end

    # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ω, Σ_ω), ω_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ω_MC_tt = ω̃
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    else
        ω_MC_tt = copy(ω_MC_lag)
        β_MC_tt = ω_MC_tt[1:k]
        α_MC_tt = ω_MC_tt[k+1]
    end
    ## 4
    println("tt=$tt, step 4")
    # (a) propose tilde from random walk
    ρ̃ = ρ_MC_lag + rand(Distributions.MvNormal(proposal_mean_ρ, proposal_var_ρ)) # multiply μ_ρ, make less variation

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            A_R_cc_temp[key] =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ̃,  ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)   
        end
        A_R[cc] = copy(A_R_cc_temp)
    end


       # for P() terms in 3(b)
    for cc in 1:num_cc
        np = 0
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            #println("$cc, $key")
            np += 1
            u_sum_A_tilde[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_lag[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m,  β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_lag[cc,np] = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_lag, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
            u_sum_A_R_tilde[cc,np] = gen_A_R_utility_sum(cc, key, A_R[cc][key], m, β_MC_tt, α_MC_tt, ρ̃, ν_MC_lag, ϕ_MC_lag, ξ_MC_lag)
        end
    end
  # for N() terms in 3(b)
    pdf_tilde = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ̃)
    pdf_lag = pdf(Distributions.MvNormal(μ_ρ, Σ_ρ), ρ_MC_lag)
    a_bar = (
            log(pdf_tilde) + sum(u_sum_A_tilde) + sum(u_sum_A_R_lag)
            -
            (log(pdf_lag) + sum(u_sum_A_lag) + sum(u_sum_A_R_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        ρ_MC_tt = ρ̃
    else
        ρ_MC_tt = copy(ρ_MC_lag)
    end

    ## 5
    println("tt=$tt, step 5")
    # (a) propose tilde from random walk
    ϕ_MC_tt = zeros(2)
    ## 6
    println("tt=$tt, step 6")
    γ_MC_tt = zeros(num_rc+2)
    ## 7
    println("tt=$tt, step 7")
    ν_temp = ν_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()
    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for ii in 1:cc_key_num_player
                # (a) propose tilde from random walk
                ν̃ = copy(ν_MC_lag)
                ν_temp_ii = ν_MC_lag[Country_List[cc]][player_rows[ii],:]
                ν̃_ii =  ν_temp_ii + rand(Normal(0,1),num_rc)
                ν̃[Country_List[cc]][player_rows[ii],:] = ν̃_ii 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_lag_7 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_7 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_7 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν̃, ϕ_MC_tt, ξ_MC_lag)
                # for N() terms in (c)
                pdf_tilde = prod(pdf(Normal(0,1), ν̃_ii[pp]) for pp in 1:num_rc)
                pdf_lag = prod(pdf(Normal(0,1), ν_temp_ii[pp]) for pp in 1:num_rc)
                # for μ terms in (c)
                μ_log_sum_tilde_7 =  gen_μ_log_sum(cc, key, m,  γ_MC_tt,ν̃)
                μ_log_sum_lag_7 =  gen_μ_log_sum(cc, key, m, γ_MC_tt,ν_MC_lag)
                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_7 + u_sum_A_R_lag_7 + μ_log_sum_tilde_7
                    -
                    (log(pdf_lag) + u_sum_A_lag_7 + u_sum_A_R_tilde_7 + μ_log_sum_lag_7)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ν_MC_lag[Country_List[cc]][player_rows[ii],:] = copy(ν̃_ii)
                end
            end
        end
    end
    ν_MC_tt = copy(ν_MC_lag)
    ## 8
    println("tt=$tt, step 8")
    #(a)
    σ̃_ηξ = rand(TruncatedNormal(σ_ηξ_MC_lag,0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2))) # make less variation
    #(b)
    # for N() terms in 3(b)
    pdf_trunc_tilde = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    pdf_trunc_lag = pdf(TruncatedNormal(μ_σ_ηξ, Σ_σ_ηξ, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)
    for cc in 1:num_cc
        for bb in 1:num_games
            mean_N_tilde = σ̃_ηξ  * τ_η_MC_tt * η_MC_tt[cc,bb]
            std_N_tilde = (1 - σ̃_ηξ^2 *  τ_η_MC_tt)^(1/2)
            pdf_N_tilde[cc,bb] = pdf(Normal(mean_N_tilde, std_N_tilde) , ξ_MC_lag[cc,bb])
            mean_N_lag = σ_ηξ_MC_lag  * τ_η_MC_tt * η_MC_tt[cc,bb]
            if (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt >= 0.0)
                std_N_lag = (1 - σ_ηξ_MC_lag^2 *  τ_η_MC_tt)^(1/2)
                pdf_N_lag[cc,bb] = pdf(Normal(mean_N_lag, std_N_lag) , ξ_MC_lag[cc,bb])
            else
                pdf_N_lag[cc,bb] = 1.0 # normalize when σ_ηξ_MC_lag is out of boundary
            end
        end
    end
    q_tilde_lag = pdf(TruncatedNormal(σ_ηξ_MC_lag, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ̃_ηξ)
    q_lag_tilde = pdf(TruncatedNormal(σ̃_ηξ, 0.5, (- τ_η_MC_tt^(-1/2)),  τ_η_MC_tt^(-1/2)), σ_ηξ_MC_lag)

    a_bar = (
                log(pdf_trunc_tilde) + sum(log.(pdf_N_tilde)) + log(q_tilde_lag)
                -
                 (log(pdf_trunc_lag) + sum(log.(pdf_N_lag)) + log(q_lag_tilde))
            )
    a_rate = log(rand(1)[1])
    if a_rate < a_bar
        σ_ηξ_MC_tt = σ̃_ηξ
    else
        σ_ηξ_MC_tt = copy(σ_ηξ_MC_lag)
    end

    ## 9 ξ
    println("tt=$tt, step 9")
    ξ_temp = ξ_MC_lag # inital value

    A_R = Dict{Int64, Dict{Int64, Matrix{Float64}}}() # first Int64: country, Int64: partition key
    A_R_cc_temp =  Dict{Int64, Matrix{Float64}}()

    for cc in 1:num_cc
        for key in collect(keys(m.partition_index[Country_List[cc]]))[1:n_par]
            player_rows = m.partition_index[Country_List[cc]][key]
            cc_key_num_player = length(player_rows)
            for bb in 1:num_games
                # (a) propose tilde from random walk
                ξ̃ = copy(ξ_MC_lag) 
                ξ_temp_bb = ξ_MC_lag[cc, bb]
                ξ̃_bb =  ξ_temp_bb + rand(Normal(0,1))
                ξ̃[cc,bb] = ξ̃_bb 
                # (b) Sim A_m^R
                A_R =  MC_A(R,cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)   
                # (c) for P() terms in (c)
                u_sum_A_tilde_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                u_sum_A_R_lag_9 = gen_A_R_utility_sum(cc, key, A_R, m,  β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_lag_9 = gen_A_utility_sum(cc, key, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_lag, ϕ_MC_tt, ξ_MC_lag)
                u_sum_A_R_tilde_9 = gen_A_R_utility_sum(cc, key, A_R, m, β_MC_tt, α_MC_tt, ρ_MC_tt, ν_MC_tt, ϕ_MC_tt, ξ̃)
                # for N() terms in (c)
                pdf_tilde = pdf(Normal(0,1), ξ̃_bb) 
                pdf_lag = pdf(Normal(0,1), ξ_temp_bb)

                a_bar = (
                    log(pdf_tilde) + u_sum_A_tilde_9 + u_sum_A_R_lag_9
                    -
                    (log(pdf_lag) + u_sum_A_lag_9 + u_sum_A_R_tilde_9)
                    )
                a_rate = log(rand(1)[1])
                if a_rate < a_bar # accept new ν_ii
                    ξ_MC_lag[cc,bb] = copy(ξ̃_bb)
                end
            end
        end
    end
    ξ_MC_tt = copy(ξ_MC_lag)

    τ_η_MC[tt] = τ_η_MC_tt
    δ_MC[tt,:] = δ_MC_tt
    ω_MC[tt,:] = ω_MC_tt
    β_MC[tt,:] = β_MC_tt
    α_MC[tt] = α_MC_tt
    ρ_MC[tt,:] = ρ_MC_tt
    ϕ_MC[tt,:] = ϕ_MC_tt
    γ_MC[tt,:] = γ_MC_tt
    ν_MC[tt] = ν_MC_tt
    ξ_MC[tt,:,:] = ξ_MC_tt
    σ_ηξ_MC[tt] = σ_ηξ_MC_tt
    η_MC[tt,:,:] = η_MC_tt
end





## save results
#=
writedlm("τ_η_MC3000_nonet.txt", τ_η_MC[1:3000], ',')
writedlm("δ_MC3000_nonet.txt", δ_MC[1:3000,:], ',')
writedlm("ω_MC3000_nonet.txt", ω_MC[1:3000,:], ',')
writedlm("β_MC3000_nonet.txt", β_MC[1:3000,:], ',')
writedlm("α_MC3000_nonet.txt", α_MC[1:3000], ',')
writedlm("ρ_MC3000_nonet.txt", ρ_MC[1:3000,:], ',')
writedlm("ϕ_MC3000_nonet.txt", ϕ_MC[1:3000,:], ',')
writedlm("γ_MC3000_nonet.txt", γ_MC[1:3000,:], ',')
writedlm("σ_ηξ_MC3000_nonet.txt", σ_ηξ_MC[1:3000], ',')

writedlm("τ_η_MC3000_10000_nonet.txt", τ_η_MC[3000:10000], ',')
writedlm("δ_MC3000_10000_nonet.txt", δ_MC[3000:10000,:], ',')
writedlm("ω_MC3000_10000_nonet.txt", ω_MC[3000:10000,:], ',')
writedlm("β_MC3000_10000_nonet.txt", β_MC[3000:10000,:], ',')
writedlm("α_MC3000_10000_nonet.txt", α_MC[3000:10000], ',')
writedlm("ρ_MC3000_10000_nonet.txt", ρ_MC[3000:10000,:], ',')
writedlm("ϕ_MC3000_10000_nonet.txt", ϕ_MC[3000:10000,:], ',')
writedlm("γ_MC3000_10000_nonet.txt", γ_MC[3000:10000,:], ',')
writedlm("σ_ηξ_MC3000_10000_nonet.txt", σ_ηξ_MC[3000:10000], ',')
=#
# read results
τ_η_MC[1:3000] = readdlm("τ_η_MC3000_nonet.txt", ',')
δ_MC[1:3000,:] = readdlm("δ_MC3000_nonet.txt", ',')
ω_MC[1:3000,:] = readdlm("ω_MC3000_nonet.txt", ',')
β_MC[1:3000,:] = readdlm("β_MC3000_nonet.txt", ',')
α_MC[1:3000] = readdlm("α_MC3000_nonet.txt", ',')
ρ_MC[1:3000,:] = readdlm("ρ_MC3000_nonet.txt", ',')
ϕ_MC[1:3000,:] = readdlm("ϕ_MC3000_nonet.txt", ',')
γ_MC[1:3000,:] = readdlm("γ_MC3000_nonet.txt", ',')
σ_ηξ_MC[1:3000] = readdlm("σ_ηξ_MC3000_nonet.txt", ',')


τ_η_MC[3000:10000] = readdlm("τ_η_MC3000_10000_nonet.txt", ',')
δ_MC[3000:10000,:] = readdlm("δ_MC3000_10000_nonet.txt", ',')
ω_MC[3000:10000,:] = readdlm("ω_MC3000_10000_nonet.txt", ',')
β_MC[3000:10000,:] = readdlm("β_MC3000_10000_nonet.txt", ',')
α_MC[3000:10000] = readdlm("α_MC3000_10000_nonet.txt", ',')
ρ_MC[3000:10000,:] = readdlm("ρ_MC3000_10000_nonet.txt", ',')
ϕ_MC[3000:10000,:] = readdlm("ϕ_MC3000_10000_nonet.txt", ',')
γ_MC[3000:10000,:] = readdlm("γ_MC3000_10000_nonet.txt", ',')
σ_ηξ_MC[3000:10000] = readdlm("σ_ηξ_MC3000_10000_nonet.txt", ',')



# price
plot(α_MC[1:10000])
price_no_net = zeros(8)
price_no_net[1] = mean(α_MC[5001:10000])
price_no_net[2] = std(α_MC[5001:10000])
price_no_net[3] = quantile(α_MC[5001:10000], 0.95)
price_no_net[4] = quantile(α_MC[5001:10000], 0.05)
price_no_net[5] = quantile(α_MC[5001:10000], 0.975)
price_no_net[6] = quantile(α_MC[5001:10000], 0.025)
price_no_net[7] = quantile(α_MC[5001:10000], 0.995)
price_no_net[8] = quantile(α_MC[5001:10000], 0.005)

# Rating
plot(β_MC[1:10000,3])
rating_no_net = zeros(8)
rating_no_net[1] = mean(β_MC[5001:10000,3])
rating_no_net[2] = std(β_MC[5001:10000,3])
rating_no_net[3] = quantile(β_MC[5001:10000,3], 0.95)
rating_no_net[4] = quantile(β_MC[5001:10000,3], 0.05)
rating_no_net[5] = quantile(β_MC[5001:10000,3], 0.975)
rating_no_net[6] = quantile(β_MC[5001:10000,3], 0.025)
rating_no_net[7] = quantile(β_MC[5001:10000,3], 0.995)
rating_no_net[8] = quantile(β_MC[5001:10000,3], 0.005)

# Multiplayer
is_mul_no_net = zeros(8)
is_mul_no_net[1] = mean(β_MC[5001:10000,2])
is_mul_no_net[2] = std(β_MC[5001:10000,2])
is_mul_no_net[3] = quantile(β_MC[5001:10000,2], 0.95)
is_mul_no_net[4] = quantile(β_MC[5001:10000,2], 0.05)
is_mul_no_net[5] = quantile(β_MC[5001:10000,2], 0.975)
is_mul_no_net[6] = quantile(β_MC[5001:10000,2], 0.025)
is_mul_no_net[7] = quantile(β_MC[5001:10000,2], 0.995)
is_mul_no_net[8] = quantile(β_MC[5001:10000,2], 0.005)

# Age

age_no_net = zeros(8)
age_no_net[1] = mean(β_MC[5001:10000,4])
age_no_net[2] = std(β_MC[5001:10000,4])
age_no_net[3] = quantile(β_MC[5001:10000,4], 0.95)
age_no_net[4] = quantile(β_MC[5001:10000,4], 0.05)
age_no_net[5] = quantile(β_MC[5001:10000,4], 0.975)
age_no_net[6] = quantile(β_MC[5001:10000,4], 0.025)
age_no_net[7] = quantile(β_MC[5001:10000,4], 0.995)
age_no_net[8] = quantile(β_MC[5001:10000,4], 0.005)


