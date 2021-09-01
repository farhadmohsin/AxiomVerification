import prefpy_io
import structures_py3
import math
import os
import itertools
# from .preference import Preference
from numpy import *
from profile import Profile

import config
from mechanism import *
import glob
import csv
from generation import *
import mov
# from mov import *
from preference import Preference
import matplotlib.pyplot as plt
#from sklearn.neural_network import *
#from sklearn.preprocessing import MinMaxScaler
#from sklearn import svm

import warnings
warnings.filterwarnings("ignore")

def read_Xdata(inputfile):
    N = 4000
    dim = 200
    X = zeros([N, dim])
    for i in range(N):
        X[i] = inputfile.readline().strip().split()
    return X

def read_Ydata(inputfile):
    N = 10000
    dim = 10
    Y = zeros([N, dim], dtype=int)
    for i in range(N):
        Y[i] = inputfile.readline().strip().split()
    return Y
def encodeY(y):
    c = 0
    for i in range(len(y)):
        c += y[i]*(2**i)
    return c

def wmg_from_strict_order(order):
    m = len(order)
    ranks = []
    for i in range(m):
        ranks.append(order.index(i+1))
    wmg = dict()
    for i in range(m):
        wmap = dict()
        for j in range(m):
            wmap[j] = 0
            if ranks[i] < ranks[j]:
                wmap[j] = 1
            if ranks[i] > ranks[j]:
                wmap[j] = -1
        wmg[i]= wmap
    return wmg

def comp_soc_file_compact(n, rules, setup):
    k = len(rules)
    winner_vector = []
    winners = []

    m = setup["m"]

    for i in range(k):
        temp = []
        for j in range(m):
            temp.append(0)
        winner_vector.append(temp)


    for i in range(len(rules)):
        winners.append([])

    file_name = get_profile_name(batch_num, m, n, setup["computed"] + 1, setup["generated"])
    inf = open(file_name, 'r')
    elections = json.load(inf)
    cmap = dict()
    for i in range(m):
        cmap[i] = "c" + str(i)
    for prof in elections:

        profile = Profile(cmap, preferences=[])
        for vote in prof:
            rank_map = dict()
            for cand in range(m):
                rank_map[cand] = vote[1:].index(cand+1)

            wmg = profile.genWmgMapFromRankMap(rank_map)
            profile.preferences.append(Preference(wmg, vote[0]))

        plu = MechanismPlurality().getWinners(profile)
        winner_vector[0][len(plu) - 1] += 1
        winners[0].append(plu)

        borda = MechanismBorda().getWinners(profile)
        winner_vector[1][len(borda) - 1] += 1
        winners[1].append(borda)

        veto = MechanismVeto().getWinners(profile)
        winner_vector[2][len(veto) - 1] += 1
        winners[2].append(veto)

        STVwinners = MechanismSTV().STVwinners(profile)
        winner_vector[3][len(STVwinners) - 1] += 1
        winners[3].append(STVwinners)

        coombs = MechanismCoombs().getWinners(profile)
        winner_vector[4][len(coombs) - 1] += 1
        winners[4].append(coombs)

        mminstance = MechanismMaximin()
        maximin = mminstance.getWinners(profile)
        winner_vector[5][len(maximin) - 1] += 1
        winners[5].append(maximin)

        schins = MechanismSchulze()
        sch = schins.getWinners(profile)
        winner_vector[6][ len(sch) - 1] += 1
        winners[6].append(sch)

        RPwinners = MechanismRankedPairs().getWinners(profile)
        # if len(RPwinners) > 1:
        # print(inputfile + " winners" + str(RPwinners) + ", WMG:" + str(profile.getWmg()))
        winner_vector[7][len(RPwinners) - 1] += 1
        winners[7].append(RPwinners)

        cpins = MechanismCopeland(0.5)
        cp = cpins.getWinners(profile)
        winner_vector[8][len(cp) - 1] += 1
        winners[8].append(cp)

    for i in range(len(rules)):
        result_file_name = results_path(batch_num) + rules[i] + "-" + str(setup["computed"]) + "-" + str(setup["generated"]) + ".json"
        inf = open(result_file_name, 'w+')
        json.dump(winners[i],inf)
    return winner_vector


def comp_winner_dist_compact(batch_num):
    rules = ['plurality', 'borda', 'veto', 'stv', 'coombs', 'maximin', 'schulze', 'rp', 'copeland']
    winner_array = []
    k = len(rules)
    setup = read_setup(batch_num)
    m = setup["m"]
    for i in range(k):
        temp = []
        for j in range(m):
            temp.append([])
        winner_array.append(temp)

    for n in range(setup["nmin"], setup["nmax"] + 1, setup["ind"]):
        file_name = get_profile_name(batch_num, m, n, setup["computed"]+1, setup["generated"])
        print("computing " + file_name)
        win = comp_soc_file_compact(n, rules, setup)
        for j in range(len(win[0])):
            for l in range(k):
                winner_array[l][j].append(win[l][j])
    print(winner_array)
    save_Results(winner_array, batch_num,  setup["computed"]+1, setup["generated"], rules)

def comp_soc_folder(n, setup, rules, rulelist):
    k = len(rules)
    winner_vector = []
    winners = []

    m = setup["m"]
    folder_name = config.data_folder + str(batch_num) + "/" + 'M' + str(setup["m"]) + 'N' + str(n) + '-soc/'
    results_folder = folder_name + "results/"

    for i in range(k):
        temp = []
        for j in range(m):
            temp.append(0)
        winner_vector.append(temp)

    os.chdir(folder_name)

    for i in range(len(rules)):
        winners.append([])

    for trial_num in range(setup["computed"],setup["generated"],1):
        inputfile = 'M'+str(m)+'N'+str(n)+'-'+str(trial_num+1)+'.soc'
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        #print("computing profile:" + inputfile)
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        print("computing: "+ inputfile)
        for i in range(len(rulelist)):
            win = rulelist[i].getWinners(profile)
            if len(win) == 0:
                exit(100)
            else:
                winner_vector[i][ len(win) -1] += 1
                winners[i].append(win)

    for i in range(len(rules)):
        result_file_name = results_path(batch_num) + 'M' + str(setup["m"]) + 'N' + str(n) + rules[i] + "-" + str(setup["computed"]) + "-" + str(setup["generated"]) + ".json"
        inf = open(result_file_name, 'w+')
        json.dump(winners[i],inf)
    return winner_vector

def sorted_directories(path):
    name_list = os.listdir(path)
    full_list = [os.path.join(path, i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)

    # if you want just the filenames sorted, simply remove the dir from each
    sorted_filename_list = [os.path.basename(i) for i in time_sorted_list ]
    filtered = []
    for i in sorted_filename_list:
        if i[0] != '.': # filter MACOS .DS files
            filtered.append(i)
    return filtered

def results_path(batch_num):
    return config.data_folder+ str(batch_num) + "/results/"

def save_Results(winner_array,batch_num, start, end, rules):
    os.chdir(results_path(batch_num))
    '''for i in range(len(rules)):
        with open('Num' + str(len(winner_array[0])) + '-' + rules[i]+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            print("saving: "+ rules[i]+str(winner_array[i]))
            writer.writerows(winner_array[i])
            f.close()'''
    with open(str(batch_num) + "-" + str(start) + "-" + str(end) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(winner_array)
        f.close()

    with open(str(batch_num) + "-" + str(start) + "-" + str(end) + '.json', 'w') as f:
        json.dump(winner_array,f)
        f.close()

def save_Results2(winner_array, rules):

    with open('Num' + str(len(winner_array[0])) + '-plu-borda.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(winner_array)
        f.close()


def comp_winner_dist(batch_num, rules, rulelist):
    #rules = ['plurality', 'borda', 'veto', 'stv', 'coombs', 'maximin', 'schulze', 'rp', 'copeland']
    winner_array = []
    k = len(rules)
    setup = read_setup(batch_num)
    m = setup["m"]
    for i in range(k):
        temp = []
        for j in range(m):
            temp.append([])
        winner_array.append(temp)

    for n in range(setup["nmin"], setup["nmax"] + 1, setup["ind"]):
        folder_name = 'M'+str(setup["m"])+'N'+str(n)+'-soc'
        print("computing " + folder_name)
        win = comp_soc_folder(n, setup, rules, rulelist)
        for j in range(len(win[0])):
            for l in range(k):
                winner_array[l][j].append(win[l][j])
    #print(winner_array)
    save_Results(winner_array, batch_num,  setup["computed"]+1, setup["generated"], rules)
    combine_csv(batch_num)

def comp_num_ties_soc_folder(pname, foldername, rules, rulelist):

    os.chdir(pname+"/"+foldername)
    k = len(rules)
    winner_vector = []
    for i in range(k):
        temp = []
        for j in range(2):
            temp.append(0)
        winner_vector.append(temp)

    filenames = glob.glob("*.soc")
    for inputfile in filenames:
        if 'results/' + inputfile + '.json' in glob.glob("results/*"):
            print(inputfile + "has been computed")
            f = open('results/' + inputfile + '.json', "r")
            data = json.load(f)
            f.close()
            for i in range(len(rulelist)):
                winner_vector[i][min(1, len(data[i]) - 1)] += 1
            continue
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)
        print("computing" +   " name" + inputfile + " m=" + str(profile.numCands) + " n=" + str(len(profile.preferences)))

        winners = []

        for i in range(len(rulelist)):
            win = rulelist[i].getWinners(profile)
            winner_vector[i][min(1, len(win) - 1)] += 1
            winners.append(win)


        with open('results/'+ inputfile +'.json', 'w', newline='') as f:
            json.dump(winners, f)
            f.close()


    with open('results/Any-ties.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(winner_vector)
        f.close()
    with open('results/Any-ties.json', 'w', newline='') as f:
        json.dump(winner_vector,f)
        f.close()
    return winner_vector

def comp_soc_folder_num_ties_2(pname, foldername, rules):
    k = len(rules)
    winner_vector = []
    for i in range(k):
        temp = []
        for j in range(2):
            temp.append(0)
        winner_vector.append(temp)

    filenames = glob.glob(pname+"/"+foldername+"/*")
    for inputfile in filenames:
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)
        print("computing profile:" + inputfile)
        plu = MechanismPlurality().getWinners(profile)
        winner_vector[0][min(1, len(plu)-1)] += 1

        borda = MechanismBorda().getWinners(profile)
        winner_vector[1][min(1, len(borda) - 1)] += 1

    return winner_vector

def comp_num_ties(data_folder, sub_folders,  rules):
    winner_array = []
    k = len(rules)
    for i in range(k):
        temp = []
        for j in range(2):
            temp.append([])
        winner_array.append(temp)
    for i in range(len(sub_folders)):
        print("computing " + sub_folders[i])
        win = comp_num_ties_soc_folder(data_folder, sub_folders[i], rules)
        for j in range(len(win[0])):
            for l in range(k):
                winner_array[l][j].append(win[l][j])
    print(winner_array)
    save_Results(winner_array, rules)

def comp_num_ties2(data_folder,  sub_folders,  rules):
    winner_array = []
    k = len(rules)
    for i in range(k):
        temp = []
        for j in range(2):
            temp.append([])
        winner_array.append(temp)
    for i in range(len(sub_folders)):
        print("computing " + sub_folders[i])
        win = comp_soc_folder_num_ties_2(data_folder, sub_folders[i], rules)
        for j in range(len(win[0])):
            for l in range(k):
                winner_array[l][j].append(win[l][j])
    print(winner_array)
    save_Results2(winner_array, rules)

def read_setup(batch_num):
    os.chdir(config.data_folder + str(batch_num))
    infile = open("setup.json")
    setup = json.load(infile)
    infile.close()
    return setup

def write_setup(batch_num, setup):
    infile = open(config.data_folder + str(batch_num) + "/setup.json", 'w')
    json.dump(setup,infile)
    infile.close()
    return setup

def generate_prefs(batch_num, trials):

    setup = read_setup(batch_num)

    for n in range(setup["nmin"], setup["nmax"] + 1, setup["ind"]):
        print("generating: n=" + str(n))
        strict_order(trials, setup["m"], n, True, 1).generation_patch(batch_num,setup["generated"])

    setup["generated"] += trials
    os.chdir(config.data_folder + str(batch_num))
    outfile = open("setup.json", 'w+')
    json.dump(setup, outfile)
    outfile.close()


def get_profile_name(batch_num, m,n, start, end):
    file_name = str(batch_num) + '-M' + str(m) + 'N' + str(n) + '-' + str(start) + "-" + str(end) + '.json'
    return file_name

def generate_prefs_compact(batch_num, trials):

    setup = read_setup(batch_num)
    m = setup["m"]

    os.chdir(config.data_folder)
    if str(batch_num) not in glob.glob("*"):
        os.mkdir(str(batch_num))
    os.chdir(str(batch_num))

    for n in range(setup["nmin"], setup["nmax"] + 1, setup["ind"]):
        print("generating: n=" + str(n))
        profiles = []
        for j in range(trials):
            pref = dict()
            for i in range(n):
                c = tuple(random.sample(list(range(1, m + 1)), m))
                if c in pref.keys():
                    pref[c] += 1
                else:
                    pref[c] = 1
            profiles.append(pref)

        new_profiles = []
        for pref in profiles:
            new_pref = []
            for ranking in pref:
                rank_list = list(ranking)
                rank_list.insert(0,pref[ranking])
                new_pref.append(rank_list)
            new_profiles.append(new_pref)
        file_name = get_profile_name(batch_num, m, n, setup["generated"]+1, setup["generated"]+trials)
        f = open(file_name, 'w+')
        json.dump(new_profiles, f)
        f.close()

    setup["generated"] += trials
    os.chdir(config.data_folder + str(batch_num))
    outfile = open("setup.json", 'w+')
    json.dump(setup, outfile)
    outfile.close()

def read_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()
    return data

def calculate_Preflib(filename):
    data = read_csv(filename)
    perc = []
    n = int(data[0][0].strip("[]")) + int(data[0][1].strip("[]"))
    for row in data:
        perc.append(round(int(row[1].strip("[]"))/n*100,1))
    return perc

def combine_csv(batch_num):
    os.chdir(results_path(batch_num))
    filelist = glob.glob('*.csv')
    #filelist.remove("Ties-combined.csv")
    data = read_csv(filelist[0])
    #print(filelist[0])
    m = len(data[0])
    #print(m)
    #print(data[0])
    n = len(data[0][0].split(","))
    combined = []
    for i in range(9):
        row = []
        for j in range(m):
            ar = []
            for l in range(n):
                ar.append(0)
            row.append(ar)
        combined.append(row)

    if 'Ties-combined.csv' in filelist:
        filelist.remove('Ties-combined.csv')
    for filename in filelist:
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            f.close()
            for i in range(9):
                for j in range (m):
                    vec = str(data[i][j]).replace("[","").replace("]","").split(",")
                    for l in range(n):
                        combined[i][j][l] += int(vec[l])
    with open('Ties-combined.csv', 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(combined)
        f.close()
    with open('Ties-combined.json', 'w+', newline='') as f:
        json.dump(combined,f)
        f.close()

def read_soc_file(inputfile):
    inf = open(inputfile, 'r')
    cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
    inf.close()
    profile = Profile(cmap, preferences=[])
    Profile.importPreflibFile(profile, inputfile)
    return profile

def figure_out_RP_Ties():
    os.chdir('your choice')
    files = glob.glob("*.soc")
    info = []
    for inputfile in files:
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)
        print("computing profile:" + inputfile)

        rpwinner = MechanismRankedPairs().getWinners(profile)
        print(rpwinner)
        if len(rpwinner) == 4:
            wmg = profile.getWmg()
            print(inputfile + wmg)
            info.append([inputfile, wmg])
    print(info)
    f = open("problematic.json", "w+")
    json.dump(info,f)
    f.close()


def figure_out_empty_winners():
    f = open('your choice')
    data = json.load(f)
    f.close()
    stat = [0, 0, 0, 0]
    prob = []
    for i in range(len(data)):
        winners = data[i]
        print(len(winners))
        if len(winners) == 0:
            prob.append(i)
        stat[len(winners) - 1] += 1
        if stat[3] > 0:
            print("warning")
    print(stat)
    print(prob)

def fix_RP_winners(batch_num, trials):
    setup = read_setup(batch_num)
    m = setup["m"]
    rp_array = [[] for j in range(m)]
    for n in range(setup["nmin"], setup["nmax"] + 1, setup["ind"]):
        folder_name = 'M' + str(setup["m"]) + 'N' + str(n) + '-soc'
        print("computing " + folder_name)
        os.chdir(config.data_folder + str(batch_num) + "/"+folder_name)
        winner_vector = [0 for j in range(m)]
        winner_set = []
        for trial_num in range(trials):
            inputfile = 'M' + str(m) + 'N' + str(n) + '-' + str(trial_num + 1) + '.soc'
            inf = open(inputfile, 'r')
            cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
            inf.close()
            # print("computing profile:" + inputfile)
            profile = Profile(cmap, preferences=[])
            Profile.importPreflibFile(profile, inputfile)

            rp = MechanismRankedPairs_AAAI().getWinners(profile)
            if len(rp) >0:
                winner_vector[len(rp) - 1] += 1
                winner_set.append(rp)
            else:
                exit(100)
        for j in range(m):
            rp_array[j].append(winner_vector[j])
            result_file_name = results_path(batch_num) + "M" + str(setup["m"]) + 'N' + str(n) + "-rp-0-" + str(trials) + ".json"
            inf = open(result_file_name, 'w+')
            json.dump(winner_set, inf)

    f = open(results_path(batch_num) + "Ties-combined.json")
    data = json.load(f)
    f.close()
    data[7] = rp_array
    f = open(results_path(batch_num) + "Ties-combined.json", "w+")
    json.dump(data, f)
    f.close()

    with open(results_path(batch_num)   + "Ties-combined.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

    #save_Results(winner_array, batch_num, setup["computed"] + 1, setup["generated"], rules)


def Statistics_Preflib_Data(folder):
    # return the histogram of candidates and voters
    os.chdir(folder)
    cand = []
    voters = []
    filenames = glob.glob("*.soc")
    for inputfile in filenames:
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        cand.append(len(cmap))
        voters.append(nvoters)

    f = open("Stat.json", "w+")
    json.dump([cand, voters], f)
    f.close



if __name__ == '__main__':
    Statistics_Preflib_Data('your choice')

    batch_num = int(sys.argv[1])
    trials = int(sys.argv[2])
    rules = ['plurality', 'borda', 'veto', 'stv', 'coombs', 'maximin', 'schulze', 'rp', 'copeland']
    rule_list = [MechanismPlurality(), MechanismBorda(), MechanismVeto(), MechanismSTV(), MechanismCoombs(), MechanismMaximin(), MechanismSchulze(), MechanismRankedPairs_AAAI_fixed(), MechanismCopeland(0.5)]

    #generate_prefs(batch_num, trials)
    '''comp_winner_dist(batch_num)

    setup = read_setup(batch_num)
    setup["computed"] = setup["generated"]
    write_setup(batch_num,setup)

    combine_csv(batch_num)'''



    comp_num_ties_soc_folder('your choice', "preflib-soc-fast", rules, rule_list)
    print(calculate_Preflib('your choice'))
