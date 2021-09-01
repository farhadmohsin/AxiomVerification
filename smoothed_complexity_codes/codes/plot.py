import matplotlib.pyplot as plt
import numpy as np
import os
import config
import json
import glob
import csv
import math
from mechanism import *
#from two_loop_LP_8_30 import  *
from ties import read_soc_file

from ties import calculate_Preflib, read_csv
from ties import read_setup
import matplotlib._color_data as mcd


def OLD_plot_ties(index):
    t = np.arange(20, 201, 20)
    s = [[40461, 42832, 43972, 44705, 45289, 45572, 45981, 46179, 46435, 46507],
    [45696, 46982, 47447, 47798, 48013, 48229, 48381, 48492, 48555, 48614],
    [38965, 42025, 43499, 44313, 44938, 45415, 45654, 45950, 46135, 46218],
    [34675, 37639, 40089, 41652, 42043, 43109, 43496, 43846, 44182, 44608],
    [33774, 39102, 40464, 41346, 42678, 43151, 43348, 44115, 44264, 44403],
    [37011, 40487, 42129, 43104, 43832, 44544, 44859, 45249, 45424, 45535],
    [37022, 40505, 42148, 43132, 43849, 44562, 44887, 45252, 45438, 45551],
    [37924, 41590, 43183, 44034, 44676, 45386, 45684, 45970, 46132, 46302],
    [41810, 42020, 42044, 42009, 42123, 42199, 42095, 42136, 42030, 41858]]

    news = []
    for row in s:
        new_row = []
        for j in row:
            new_row.append((1-j/50000)*100)
        news.append(new_row)

    rules = ['Plurality', 'Borda', 'Veto', 'STV','Coombs', 'Maximin','Schulze','Ranked pairs','Copeland']
    mk = ["o", "v", "s", "*", "P", "^", "X", "<", ">"]


    sub_news = [news[i] for i in index]
    sub_rules = [rules[i] for i in index]

    print(sub_news)
    for i in range(len(index)):
        plt.plot(t, sub_news[i], marker = mk[index[i]], markersize=10, color = "C"+str(index[i]))
    plt.legend(sub_rules, loc ="upper right")
    plt.xlabel('n')
    plt.xticks(range(20, 201,20), range(20, 201,20))
    plt.ylabel('% of tied profiles')
    #plt.title('About as simple as it gets, folks')
    plt.grid(True)
    #plt.savefig("test.png")
    plt.show()


def plot_Cond(batch_num,min, max, ind):
    os.chdir(config.data_folder + str(batch_num) + "/sat_results/")
    t = np.arange(min, max +1, ind)
    f = open("Condorcet-combined.json")
    s_cond = json.load(f)
    f.close


    trials = s_cond[0][0]
    news = []
    for row in s_cond:
        new_row = []
        for j in range(len(row)):
            new_row.append(row[j]*100/s_cond[8][j])
        news.append(new_row)

    rules = ['Plurality', 'Borda', 'Veto', 'STV','Black', 'Maximin','Schulze','Ranked pairs','Copeland']
    mk = ["o", "v", "s", "*", "P", "^", "X", "<", ">"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "tab:brown", "tab:grey"]
    index = [0,1, 2, 3]

    sub_news = [news[i] for i in index]
    sub_rules = [rules[i] for i in index]

    print(sub_news)
    for i in range(len(index)):
        plt.plot(t, sub_news[i], marker=mk[index[i]], markersize=10, color = colors[index[i]])
    plt.legend(sub_rules, loc="center right")
    plt.xlabel('n')
    plt.xticks(range(min, max +1, 2*ind), range(min, max +1, 2*ind))
    plt.ylabel('% satisfaction of Condorcet')
    # plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig(str(batch_num) + "-Condorcet.png")
    plt.show()



def plot_Par(batch_num, min, max, ind, index = range(3, 9)):
    os.chdir(config.data_folder + str(batch_num) + "/sat_results/")
    t = np.arange(min, max +1, ind)

    f = open("Participation-combined.json")
    s_par = json.load(f)
    f.close

    trials = s_par[0][0]
    news = []
    for row in s_par:
        new_row = []
        for j in range(len(row)):
            new_row.append(row[j]*100/s_par[0][j])
        news.append(new_row)
    colors = ["b", "g", "r", "c", "m", "y", "k", "tab:brown", "tab:grey"]
    rules = ['Plurality', 'Borda', 'Veto', 'STV','Black', 'Maximin','Schulze','Ranked pairs','Copeland']
    mk = ["o", "v", "s", "*", "P", "^", "X", "<", ">"]


    sub_news = [news[i] for i in index]
    sub_rules = [rules[i] for i in index]

    for i in range(len(index)):
        plt.plot(t, sub_news[i], marker=mk[index[i]], markersize=10, color = colors[index[i]]) #"C"+str(index[i])
    plt.legend(sub_rules, loc="lower right")
    plt.xlabel('n')
    plt.xticks(range(min, max +1, 2*ind), range(min, max +1, 2*ind))
    plt.ylabel('% satisfaction of Participation')
    # plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig(str(batch_num) + "-Participation.png")
    plt.show()


def plot_Cond_merged(batch_num, min, max, ind):
    os.chdir(config.data_folder + str(batch_num) + "/sat_results/merged")
    t = np.arange(min, max+1, ind)
    f = open("N" + str(min) + "-" + str(max) + "-Condorcet-combined.json")
    s_cond = json.load(f)
    f.close


    trials = s_cond[0][0]
    news = []
    for row in s_cond:
        new_row = []
        for j in range(len(row)):
            new_row.append(row[j]*100/s_cond[8][j])
        news.append(new_row)

    rules = ['Plurality', 'Borda', 'Veto', 'STV','Black', 'Maximin','Schulze','Ranked pairs','Copeland']
    mk = ["o", "v", "s", "*", "P", "^", "X", "<", ">"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "tab:brown", "tab:grey"]
    index = [0,1, 2, 3]

    sub_news = [news[i] for i in index]
    sub_rules = [rules[i] for i in index]

    print(sub_news)
    for i in range(len(index)):
        plt.plot(t, sub_news[i], marker=mk[index[i]], markersize=10, color = colors[index[i]])
    plt.legend(sub_rules, loc="center right")
    plt.xlabel('n')
    plt.xticks(range(min, max +1, ind*2), range(min, max+1, ind*2))
    plt.ylabel('% satisfaction of Condorcet')
    plt.grid(True)
    plt.savefig(str(batch_num) + "-Condorcet.png")
    plt.show()
    return plt

def inverse_sqrt(input):
    return  [1/math.sqrt(i) for i in input]

def plot_Par_merged(batch_num, min, max, ind):
    os.chdir(config.data_folder + str(batch_num) + "/sat_results/merged/")
    t = np.arange(min, max+1, ind)

    f = open("N" + str(min) + "-" + str(max) + "-Participation-combined.json")
    s_par = json.load(f)
    f.close


    trials = s_par[0][0]
    news = []
    for row in s_par:
        new_row = []
        for j in range(len(row)):
            new_row.append(row[j]*100/s_par[0][j])
        news.append(new_row)
    colors = ["b", "g", "r", "c", "m", "y", "k", "tab:brown", "tab:grey"]
    rules = ['Plurality', 'Borda', 'Veto', 'STV','Black', 'Maximin','Schulze','Ranked pairs','Copeland']
    mk = ["o", "v", "s", "*", "P", "^", "X", "<", ">"]
    index = range(3, 9)

    sub_news = [news[i] for i in index]
    sub_rules = [rules[i] for i in index]

    for i in range(len(index)):
        plt.plot(t, sub_news[i], marker=mk[index[i]], markersize=10,  color = colors[index[i]]) #"C"+str(index[i])
        '''y = np.array(sub_news[i])
        fit = np.poly1d(np.polyfit(inverse_sqrt(t), y, 1))
        x = np.linspace(40, 800, 100)
        plt.plot(x, fit(inverse_sqrt(x)), "--", color = colors[index[i]])'''
    plt.legend(sub_rules, loc="lower right")




    plt.xlabel('n')
    plt.xticks(range(min, max +1, ind*2), range(min, max+1, ind*2))
    plt.ylabel('% satisfaction of Participation')
    # plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig(str(batch_num) + "-Participation.png")
    plt.show()
    return plt

def combine_Cond_Par_Json(foldername):
    os.chdir(foldername)
    files = glob.glob('results/*.json')
    f = open(files[0], 'r')
    temp = json.load(f)
    f.close()
    m = len(temp[0])
    total = [[0 for i in range(m)], [0 for i in range(m)]]
    #print(files)
    for file in files:
        f = open(file,'r')
        temp = json.load(f)
        f.close()
        for i in range(m):
            total[0][i] += temp[0][i]
            total[1][i] += temp[1][i]
    return len(files), total

def combine_One_Seven(filename):
    os.chdir('your choice')
    f = open(filename + ".json", 'r')
    data1 = json.load(f)
    f.close()
    os.chdir('your choice')
    f = open(filename + ".json", 'r')
    data2 = json.load(f)
    f.close()
    data_new = [[] for i in range(len(data1))]
    index = [1,3,5,7,9]
    for i in range(len(data1)):
        for j in index:
            data_new[i].append(data1[i][j])
        for j in range(len(data2[0])):
            data_new[i].append(data2[i][j])
    f = open("N40-400-" + filename + ".json", 'w+')
    json.dump(data_new,f)
    f.close()

def combine_Eight_Nine(filename):
    os.chdir("/Users/administrator/GGSDDU/Codes/2021-smoothed-ties/data/8/sat_results/")
    f = open(filename + ".json", 'r')
    data1 = json.load(f)
    f.close()
    os.chdir("/Users/administrator/GGSDDU/Codes/2021-smoothed-ties/data/9/sat_results/")
    f = open(filename + ".json", 'r')
    data2 = json.load(f)
    f.close()

    for i in range(len(data1)):
        data1[i].extend(data2[i])
    f = open("merged/N40-800-" + filename + ".json", 'w+')
    json.dump(data1, f)
    f.close()

def combine_One_Twelve(filename):
    os.chdir("/Users/administrator/GGSDDU/Codes/2021-smoothed-ties/data/1/sat_results/")
    f = open(filename + ".json", 'r')
    data1 = json.load(f)
    f.close()
    os.chdir("/Users/administrator/GGSDDU/Codes/2021-smoothed-ties/data/12/sat_results/")
    f = open(filename + ".json", 'r')
    data2 = json.load(f)
    f.close()

    for i in range(len(data1)):
        data1[i].extend(data2[i])
    f = open("merged/N20-400-" + filename + ".json", 'w+')
    json.dump(data1, f)
    f.close()

    with open("merged/N20-400-" + filename  + '.csv', 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data1)
        f.close()


def plot_Ties(batch_num, min, max, ind, k, trials,index):
    os.chdir(config.data_folder + str(batch_num) + "/results/")
    t = np.arange(min, max +1, ind)

    f = open("Ties-combined.json")
    data = json.load(f)
    f.close

    k_way_ties = [data[j][k-1] for j in range(len(data)) ]

    news = []
    for row in k_way_ties:
        new_row = []
        for j in range(len(row)):
            new_row.append(row[j]*100/trials)
        news.append(new_row)
    colors = ["b", "g", "r", "c", "m", "y", "k", "tab:brown", "tab:grey"]
    rules = ['Plurality', 'Borda', 'Veto', 'STV','Coombs', 'Maximin','Schulze','Ranked pairs','Copeland']
    mk = ["o", "v", "s", "*", "P", "^", "X", "<", ">"]

    sub_news = [news[i] for i in index]
    sub_rules = [rules[i] for i in index]

    for i in range(len(index)):
        plt.plot(t, sub_news[i], marker=mk[index[i]], markersize=10, color = colors[index[i]]) #"C"+str(index[i])
    plt.legend(sub_rules, loc=0)
    plt.xlabel('n')
    plt.xticks(range(min, max +1, ind), range(min, max +1, ind))
    plt.ylabel('% tied profiles')
    # plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig(str(batch_num) + "-"+ str(k)+"way-Ties.png")
    plt.show()

def plot_Any_Ties(batch_num, min, max, ind, trials,index):
    os.chdir(config.data_folder + str(batch_num) + "/results/")
    t = np.arange(min, max +1, ind)

    f = open("Ties-combined.json")
    data = json.load(f)
    f.close

    k_way_ties = [data[j][0] for j in range(len(data)) ]

    news = []
    for row in k_way_ties:
        new_row = []
        for j in range(len(row)):
            new_row.append((1-row[j]/trials)*100)
        news.append(new_row)
    colors = ["b", "g", "r", "c", "m", "y", "k", "tab:brown", "tab:grey"]
    rules = ['Plurality', 'Borda', 'Veto', 'STV','Coombs', 'Maximin','Schulze','Ranked pairs','Copeland']
    mk = ["o", "v", "s", "*", "P", "^", "X", "<", ">"]

    sub_news = [news[i] for i in index]
    sub_rules = [rules[i] for i in index]

    for i in range(len(index)):
        plt.plot(t, sub_news[i], marker=mk[index[i]], markersize=10, color = colors[index[i]]) #"C"+str(index[i])
    plt.legend(sub_rules, loc=0)
    plt.xlabel('n')
    plt.xticks(range(min, max +1, ind), range(min, max +1, ind))
    plt.ylabel('% tied profiles')
    # plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig(str(batch_num) + "-Ties.png")
    plt.show()

def fib_frog(n):
    f = [1 for i in range(n)]
    for i in range(2, n):
        f[i] = f[i - 1] + f[i - 2]
        #print("I am the " + str(f[i]) + "th frog")
    t = range(n)
    plt.plot(t, f, marker= "*", markersize=10, )
    plt.xticks(t,t)
    plt.grid(True)
    plt.show()

def plot_Stat_Preflit(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cand = data[0]
    voters = data[1]
    a = range
    plt.hist(data[0], density=False)  # density=False would make counts
    plt.ylabel('Number of profiles')
    plt.xlabel('Number of candidates');
    #plt.show()
    plt.savefig("hist_cand.png")

    plt.clf()
    plt.hist(data[1], density=False, bins= 30)  # density=False would make counts
    plt.ylabel('Number of profiles')
    plt.xlabel('Number of voters');
    #plt.show()
    plt.savefig("hist_voters.png")


if __name__ == '__main__':

    """Codes for plotting Satisfaction of axioms"""
    plot_Stat_Preflit("/Users/administrator/GGSDDU/Codes/2021-smoothed-ties/data/preflib-soc-all/Stat.json")



    """Codes for plotting Ties
    index = [1, 3, 5, 7, 8]
    index = range(9)
    #plot_ties(index)
    #plot_Ties(1, 20, 200, 20, 2, 100000, index)
    plot_Any_Ties(1, 20, 200, 20, 100000, index)"""

    profile = read_soc_file('your choice')
