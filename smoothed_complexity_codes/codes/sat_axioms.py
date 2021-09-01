import prefpy_io
from ties import read_setup, results_path, read_csv

import config
from mechanism import *
import glob
import csv
from generation import *
from preference import Preference


def save_SAT_Results(Cond_array, Par_array, batch_num, start, end):
    os.chdir(config.data_folder + str(batch_num))
    if "sat_results" not in glob.glob("*"):
        os.mkdir("sat_results")
    os.chdir("sat_results")
    fname_Condorcet = str(batch_num) + "-Condorcet-" + str(start)  + '-'+str(end)
    fname_Par = str(batch_num) + "-Participation-" + str(start)  + '-'+str(end)
    with open(fname_Condorcet  + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Cond_array)
        f.close()
    with open(fname_Condorcet  + '.json', 'w', newline='') as f:
        json.dump(Cond_array,f)
        f.close()
    with open(fname_Par + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Par_array)
        f.close()
    with open(fname_Par  + '.json', 'w', newline='') as f:
        json.dump(Par_array,f)
        f.close()

def read_soc_file(inputfile):
    inf = open(inputfile, 'r')
    cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
    inf.close()
    profile = Profile(cmap, preferences=[])
    Profile.importPreflibFile(profile, inputfile)
    return profile


def comp_Sat(batch_num, start, end, rules, rule_list):
    os.chdir(config.data_folder + str(batch_num))
    setup = read_setup(batch_num)
    Cond_final = [[] for i in rule_list]
    Par_final = [[] for i in rule_list]
    for n in range(setup["nmin"], setup["nmax"] + 1, setup["ind"]):
        folder_name = 'M'+str(setup["m"])+'N'+str(n)+'-soc'
        print("computing " + folder_name)
        Condorcet_array = [0 for i in range(len(rule_list))]
        Participation_array = [0 for i in range(len(rule_list))]
        for index in range(start, end+1):
            soc_file_name = folder_name + '/' + 'M'+str(setup["m"])+'N'+str(n)+'-'+str(index)+'.soc'
            profile = read_soc_file(soc_file_name)
            for i in range(len(rule_list)):
                Condorcet_array[i] += rule_list[i].satCondorcet(profile)
                Participation_array[i] += rule_list[i].satParticipation(profile)
        for j in range(len(rule_list)):
            Cond_final[j].append(Condorcet_array[j])
            Par_final[j].append(Participation_array[j])
    save_SAT_Results(Cond_final, Par_final, batch_num, start, end)
    return Cond_final, Par_final

def comp_Sat_folder(folder_name, rules, rule_list):
    os.chdir(config.data_folder + folder_name)
    file_list = glob.glob("*.soc")

    Cond_final = [[] for i in rule_list]
    Par_final = [[] for i in rule_list]

    Condorcet_array = [0 for i in range(len(rule_list))]
    Participation_array = [0 for i in range(len(rule_list))]

    fileind = 0
    total_num = 0
    for soc_file_name in file_list:
        fileind += 1

        if "results/" + soc_file_name + ".json" in glob.glob("results/*.json"):
            print("already computed" + soc_file_name)
            f = open("results/" + soc_file_name + ".json", "r")
            data = json.load(f)
            f.close()
            for k in range(len(data[0])):
                Condorcet_array[k] += data[0][k]
                Participation_array[k] += data[1][k]
            continue
        profile = read_soc_file(soc_file_name)
        print("computing" + str(total_num) + " name" + soc_file_name + " m=" + str(profile.numCands) + " n=" + str(len(profile.preferences)))
        index = 0
        Cond_inc = [0 for i in range(len(rule_list))]
        Par_inc = [0 for i in range(len(rule_list))]
        tic = time.perf_counter()
        for i in range(len(rule_list)):
            index += 1

            print(rules[i]+" computing Condorcet "+ str(index))
            Cond_inc[i] = rule_list[i].satCondorcet(profile)
            Condorcet_array[i] += Cond_inc[i]
            print(rules[i]+" computing Par " + str(index))
            Par_inc[i] = rule_list[i].satParticipation(profile)
            Participation_array[i] += Par_inc[i]
        toc = time.perf_counter()
        inf = open("results/" + soc_file_name + ".json" , 'w+')
        json.dump([Cond_inc, Par_inc, toc-tic], inf)
        inf.close()
    for j in range(len(rule_list)):
        Cond_final[j].append(Condorcet_array[j])
        Par_final[j].append(Participation_array[j])


    with open("Preflib-Condorcet-" + str(len(file_list))  + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Cond_final)
        f.close()
    with open("Preflib-Participation-" + str(len(file_list))  + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(Par_final)
        f.close()

    return len(file_list), Cond_final, Par_final

def combine_SAT_csv(batch_num, rule_list, axiom):
    os.chdir(config.data_folder + str(batch_num) + "/sat_results/")
    filelist = glob.glob('*' + axiom +  '*.csv')
    data = read_csv(filelist[0])
    #print(filelist[0])
    m = len(data[0])
    #print(m)
    #print(data[0])
    n = len(data[0])
    combined = []
    for i in range(len(rule_list)):
        combined.append([0 for j in range(n)])

    if axiom+'-combined.csv' in filelist:
        filelist.remove(axiom+'-combined.csv')
    for filename in filelist:
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            f.close()
            for i in range(len(rule_list)):
                #vec = str(data[i]).strip("[]'").split(",")
                for l in range(n):
                    combined[i][l] += int(data[i][l])
    with open(axiom+'-combined.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(combined)
        f.close()

    with open(axiom+'-combined.json', 'w', newline='') as f:
        json.dump(combined, f)
        f.close()


def comparefolder():
    filelist = glob.glob('your choice')
    for file in filelist:
        profile = read_soc_file(file)
        a = MechanismRankedPairs().lexWinner(profile)
        b = MechanismRankedPairs().lexWinner_old(profile)
        if a != b:
            print(file + ": " + str(a) + "vs " + str(b))

def comparefile(file):
    profile = read_soc_file(file)
    a= MechanismRankedPairs().lexWinner(profile)
    b = MechanismRankedPairs().lexWinner_old(profile)
    if a != b:
        print(file + ": " + str(a) + "vs " +str(b))
if __name__ == '__main__':


    batch_num = int(sys.argv[1])
    start = int(sys.argv[2])
    end = int(sys.argv[3])



    rules = ['Plurality', 'Borda', 'Veto', 'STV', 'Black', 'Maximin', 'Schulze', 'Ranked pairs', 'Copeland']
    rule_list = [MechanismPlurality(), MechanismBorda(), MechanismVeto(), MechanismSTV(),MechanismBlack(), MechanismMaximin(), MechanismSchulze(), MechanismRankedPairs(), MechanismCopeland(0.5)]


    for i in range(50000, 190001, 10000):
        print("starting trial "+ str(i))
        comp_Sat(batch_num, i+1, i+10000, rules, rule_list)
        combine_SAT_csv(batch_num, rule_list,"Condorcet")
        combine_SAT_csv(batch_num, rule_list, "Participation")
