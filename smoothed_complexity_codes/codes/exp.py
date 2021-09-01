import copy
import prefpy_io
import itertools
import math
import json
import os
from preference import Preference
from numpy import *
from structures_py3 import *
from profile import Profile
import mechanism
import mov
import glob

def STVwinners(ordering, prefcounts, m, startstate):
    """
    Returns an integer list that represents all winners of a profile.

    :ivar array<int> ordering: An ordering matrix that represents an election profile.
    :ivar list<int> prefcounts: An list of numbers, each of which represent a pattern of the votes.
    :ivar int n: the total number of the votes.
    :ivar int m: the total number of the candidates.
    :ivar set<int> startstate: An set of numbers that represent the candidates.
    """
    ordering, startstate = preprocessing(ordering, prefcounts, m, startstate)
    m_star = len(startstate)
    known_winners = set()
    # ----------Some statistics--------------
    branchestotal = 0
    hashtable2 = set()
    # PLUSCORETIME = 0
    pluscorenum = 0
    nodesbylemma1 = 0
    cache_hits = 0
    step = 0
    steps = []

    # push the node of start state into the priority queue
    root = Node(value=startstate)
    stackNode = []
    stackNode.append(root)

    while stackNode:
        # ------------pop the current node-----------------
        node = stackNode.pop()
        step += 1
        # print(node.value)
        # -------------------------------------------------
        state = node.value.copy()

        # use heuristic to delete all the candidates which satisfy the following condition

        # goal state 1: if the state set contains only 1 candidate, then stop
        if len(state) == 1 and list(state)[0] not in known_winners:
            known_winners.add(list(state)[0])
            steps.append(step)
            continue
        # goal state 2 (pruning): if the state set is subset of the known_winners set, then stop
        if state <= known_winners:
            nodesbylemma1 += 1
            continue
        # ----------Compute plurality score for the current remaining candidates--------------
        # PLUSTART = time.perf_counter()
        plural_score = get_plurality_scores3(prefcounts, ordering, state, m_star)
        pluscorenum += 1
        # PLUEND = time.perf_counter()
        # PLUSCORETIME += PLUEND - PLUSTART

        # if current state satisfies one of the 3 goal state, continue to the next loop

        # After using heuristics, generate children and push them into priority queue
        # frontier = [val for val in known_winners if val in state] + list(set(state) - set(known_winners))
        # childbranch = 0
        minscore = min(plural_score.values())
        for to_be_deleted in state:
            if plural_score[to_be_deleted] == minscore:
                child_state = state.copy()
                child_state.remove(to_be_deleted)
                # HASHSTART = time.perf_counter()
                tpc = tuple(sorted(child_state))
                if tpc in hashtable2:
                    cache_hits += 1
                    # HASHEND = time.perf_counter()
                    # HASHTIME += HASHEND - HASHSTART
                    continue
                else:
                    hashtable2.add(tpc)
                    # HASHEND = time.perf_counter()
                    # HASHTIME += HASHEND - HASHSTART
                    child_node = Node(value=child_state)
                    stackNode.append(child_node)
                    branchestotal += 1
    step_percentage = [i / step for i in steps]
    return sorted(known_winners), pluscorenum, branchestotal, cache_hits, nodesbylemma1, m_star, len(known_winners), step_percentage


def preprocessing(ordering, prefcounts, m, startstate):
    plural_score = get_plurality_scores3(prefcounts, ordering, startstate, m)
    state = set([key for key, value in plural_score.items() if value != 0])
    ordering = construct_ordering(ordering, prefcounts, state)
    plural_score = dict([(key, value) for key, value in plural_score.items() if value != 0])
    # plural_score = get_plurality_scores3(prefcounts, ordering, state, m)
    minscore = min(plural_score.values())
    to_be_deleted = [key for key, value in plural_score.items() if value == minscore]
    if len(to_be_deleted) > 1:
        return ordering, state
    else:
        while len(to_be_deleted) == 1 and len(state) > 1:
            state.remove(to_be_deleted[0])
            plural_score = get_plurality_scores3(prefcounts, ordering, state, m)
            minscore = min(plural_score.values())
            to_be_deleted = [key for key, value in plural_score.items() if value == minscore]
        ordering = construct_ordering(ordering, prefcounts, state)
        return ordering, state


def construct_ordering(ordering, prefcounts, state):
    new_ordering = []
    for i in range(len(prefcounts)):
        new_ordering.append([x for x in ordering[i] if x in state])
    return new_ordering


def get_plurality_scores3(prefcounts, ordering, state, m):
    # print(state)
    plural_score = {}
    plural_score = plural_score.fromkeys(state, 0)
    for i in range(len(prefcounts)):
        for j in range(m):
            if ordering[i][j] in state:
                plural_score[ordering[i][j]] += prefcounts[i]
                break
    return plural_score


def read_election_file4(inputfile):
    # first element is the number of candidates.
    l = inputfile.readline()
    m = int(l.strip())  # The number of alternatives
    bits = inputfile.readline()
    if int(bits[0].strip()) == 0:  # The name of the 1st candidate is "0"
        startstate = set(range(m))
    else:
        startstate = set(range(1, m + 1))
    for i in range(m - 1):
        bits = inputfile.readline()
    bits = inputfile.readline().strip().split(",")
    n = int(bits[0].strip())  # The total number of votes
    sumvotes = int(bits[1].strip())
    len_prefcounts = int(bits[2].strip())
    ordering = zeros([len_prefcounts, m], dtype=int)
    prefcounts = []
    for i in range(len_prefcounts):
        rec = inputfile.readline().strip()
        count = int(rec[:rec.index(",")])
        bits = rec[rec.index(",") + 1:].strip().split(",")
        ordering[i] = bits
        prefcounts.append(count)
    # Sanity check:
    if sum(prefcounts) != sumvotes or size(ordering, 0) != len_prefcounts:
        print("Error Parsing File: Votes Not Accounted For!")
        exit()

    return ordering, prefcounts, n, m, startstate


def read_election_file(inputfile):
    # first element is the number of candidates.
    l = inputfile.readline()
    numcands = int(l.strip())
    candmap = {}
    for i in range(numcands):
        bits = inputfile.readline().strip().split(",")
        candmap[int(bits[0].strip())] = bits[1].strip()

    # now we have numvoters, sumofvotecount, numunique orders
    bits = inputfile.readline().strip().split(",")
    numvoters = int(bits[0].strip())
    sumvotes = int(bits[1].strip())
    uniqueorders = int(bits[2].strip())

    rankmaps = []
    rankmapcounts = []
    for i in range(uniqueorders):
        rec = inputfile.readline().strip()
        # need to parse the rec properly..
        if rec.find("{") == -1:
            # its strict, just split on ,
            count = int(rec[:rec.index(",")])
            bits = rec[rec.index(",") + 1:].strip().split(",")
            cvote = {}
            for crank in range(len(bits)):
                cvote[int(bits[crank])] = crank + 1
            rankmaps.append(cvote)
            rankmapcounts.append(count)
        else:
            count = int(rec[:rec.index(",")])
            bits = rec[rec.index(",") + 1:].strip().split(",")
            cvote = {}
            crank = 1
            partial = False
            for ccand in bits:
                if ccand.find("{") != -1:
                    partial = True
                    t = ccand.replace("{", "")
                    cvote[int(t.strip())] = crank
                elif ccand.find("}") != -1:
                    partial = False
                    t = ccand.replace("}", "")
                    cvote[int(t.strip())] = crank
                    crank += 1
                else:
                    cvote[int(ccand.strip())] = crank
                    if partial == False:
                        crank += 1
            rankmaps.append(cvote)
            rankmapcounts.append(count)

    # Sanity check:
    if sum(rankmapcounts) != sumvotes or len(rankmaps) != uniqueorders:
        print("Error Parsing File: Votes Not Accounted For!")
        exit()

    return candmap, rankmaps, rankmapcounts, numvoters

# Below is a template Main which shows off some of the
# features of this library.
if __name__ == '__main__':

    print("------------------------------")
    os.chdir('D:\Social Choice\data\soc-3')
    # os.chdir('D:\Social Choice\Programming\soc')
    # result = open('D:\\Social Choice\\Programming\\soc-4-p0h0s0c1.txt', 'w+')

    data_range = [100]
    for M in [10]:
        # start0 = time.perf_counter()
        for N in data_range:
            filenames = glob.glob('M' + str(M) + 'N' + str(N) + '-*.soc')
            # print("ok")
            # filenames = glob.glob('ED*.soc')
            # total_time = 0
            # total_nodes = 0
            # num_profile = 0
            # total_plu = 0
            # total_pruning = 0
            # total_cache_hits = 0
            # total_ties = 0
            # hardcases = 0
            for inputfile in filenames:
                # print("ok")
                inf = open(inputfile, 'r')
                ordering, prefcounts, n, m, startstate = read_election_file4(inf)
                cmap, rmaps, rmapscounts, nvoters = read_election_file(inf)
                inf.close()
                # print("ok")
                # start = time.perf_counter()
                winners = STVwinners(ordering, prefcounts, m, startstate)

                profile = Profile(cmap, preferences=[])
                Profile.importPreflibFile(profile, inputfile)
                wmg = profile.getWmg()
                Copelandscores = mov.getCopelandScores(profile)
                stv = mechanism.MechanismSTV(profile)
                # end = time.perf_counter()
                print(winners, wmg, Copelandscores)
                # print("{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10}".format(inputfile, winners[0], (end - start), winners[1],
                #                                                winners[2], winners[3], winners[4], winners[5],
                #                                                winners[6], winners[7], winners[8]), file=result)
                # total_time += end - start
                # total_plu += winners[1]
                # total_nodes += winners[2]
                # total_cache_hits += winners[3]
                # total_pruning += winners[4]
                # total_ties += winners[8]
                # if winners[8] > 0:
                #     hardcases += 1
                # num_profile += 1
                # if num_profile >= 500:
                #     break

            # average_time = total_time / num_profile
            # average_plu = total_plu / num_profile
            # average_nodes = total_nodes / num_profile
            # average_cache_hits = total_cache_hits / num_profile
            # average_pruning = total_pruning / num_profile
            # average_ties = total_ties / num_profile
            # average_ties = total_ties / num_profile

            # print(str(N) + " %f %f %f %f %f %f %d" % (average_time, average_plu, average_nodes, average_cache_hits, average_pruning, average_ties, hardcases))
            # print("{0} {1} {2} {3} {4} {5}".format(average_time, average_plu, average_nodes, average_cache_hits,
            #                                        average_pruning, 0), file=result)

        # end0 = time.perf_counter()
        # print('M' + str(M) + ":total_time = %f s" % (end0 - start0))

    # result.close()