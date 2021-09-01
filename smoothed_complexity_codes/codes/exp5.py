import prefpy_io
import math
import os
import itertools
import time
from numpy import *
from profile import Profile
from mechanism import *
import glob
from mov import *
import numpy as np

# import prefpy
# from .preference import Preference
# from .profile import Profile
# from prefpy import preference

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
        c += y[i] * (2 ** i)
    return c


def cyclic(graph):
    """
    Return True if the directed graph has a cycle.
    The graph must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False
    >>> cyclic({1: [2], 2: [3], 3: [1]})
    True

    Gareth Rees. https://codereview.stackexchange.com/questions/86021/check-if-a-directed-graph-contains-a-cycle
    """
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [iter(graph)]
    while stack:
        for v in stack[-1]:
            if v in path_set:
                return True
            elif v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(graph.get(v, ())))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    return False


def topological_sort(graph):
    """
    Return a list corresponding the order of the vertices of a dag.
    The graph must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> topological_sort({1: (2,), 2: (3,), 3: (4,), 4:()})
    [1, 2, 3, 4]

    http://blog.csdn.net/pandora_madara/article/details/26478385
    """
    is_visit = dict((node, False) for node in graph)
    li = []

    def dfs(graph, start_node):

        for end_node in graph[start_node]:
            if not is_visit[end_node]:
                is_visit[end_node] = True
                dfs(graph, end_node)
        li.append(start_node)

    for start_node in graph:
        if not is_visit[start_node]:
            is_visit[start_node] = True
            dfs(graph, start_node)

    li.reverse()
    return li


def vectorize_wmg(wmg):
    m = len(wmg)
    n = np.sum(np.abs([wmg[0][i] for i in range(1, m)]))
    wmg_vec = [wmg[i][j] for i in range(m) for j in range(m) if not j == i]
    print("wmg_vec=", wmg_vec)
    wmg_vec_normalized = list(1. * np.array(wmg_vec) / n)
    return wmg_vec_normalized


def rankedpairs(profile):
    """
    Returns a number that associates the winner of a profile under ranked pairs rule.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates. Ties are
    # allowed however.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported election type")
        exit()

    wmg = profile.getWmg()
    m = profile.numCands
    ordering = profile.getOrderVectors()

    if min(ordering[0]) == 0:
        I = list(range(m))
    else:
        I = list(range(1, m + 1))

    G = dict()
    for i in I:
        G[i] = []
    wmg2 = dict()
    for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
        wmg2[(cand1, cand2)] = wmg[cand1][cand2]

    while wmg2 is not None:
        (edge, weight) = max(wmg2.items(), key=lambda x: x[1])
        print(edge)
        if weight < 0:
            break
        print(G[edge[0]])
        print('G1=', G)
        G[edge[0]].append(edge[1])
        print(G[edge[0]])
        print('G2=', G)
        print(cyclic(G))

        if cyclic(G) is False:
            wmg2.pop(edge)
            print("ok", G)
        else:
            G[edge[0]].remove(edge[1])
            wmg2.pop(edge)
    print('G=', G)
    top_list = topological_sort(G)
    return top_list[0]


if __name__ == '__main__':
    os.chdir('D:\Social Choice\data\soc-3-hardcase')
    # os.chdir('D:\Social Choice\data\soc-toc-jw')
    # inputfile = "ED-10004-00000025.soc"
    # inputfile = "M20N20-100001.csv"

    filenames = glob.glob('M10N10-1*.csv')
    # filenames = glob.glob('ED-10004-00000002.soc')
    filenames = filenames[0:9]
    # result = open('D:\\Social Choice\\Programming\\M10N10-100-ranked-pairs.txt', 'w+')
    num_profile = 0
    nodes = 0
    cycles = 0
    time_s = 0
    time_c = 0

    for inputfile in filenames:

        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)


        start = time.perf_counter()
        single_winner = MechanismRankedPairs().ranked_pairs_single_winner(profile)
        end1 = time.perf_counter()
        rp_cowinner, num_nodes, num_cycles = MechanismRankedPairs().ranked_pairs_cowinners(profile)
        end2 = time.perf_counter()
        print("%s: sw=%r, cw=%r (#nodes=%d, #cycles=%d); time_s=%f s, time_c=%f s" %
              (inputfile, single_winner,  rp_cowinner, num_nodes, num_cycles, (end1-start), (end2-end1)))
        # print("{0}\t{1}\t{2}\t{3}\t{4}".format(inputfile, single_winner,  rp_cowinner, (end1-start), (end2-end1)), file=result)
        num_profile += 1
        nodes += num_nodes
        cycles += num_cycles
        time_s += end1-start
        time_c += end2-end1

    aver_time_s = time_s / num_profile
    aver_time_c = time_c / num_profile
    aver_nodes = nodes / num_profile
    aver_cycles = cycles / num_profile
    print("aver_time_s=%f s, aver_time_c=%f s, aver_nodes=%f, aver_cycles=%f."
          % (aver_time_s, aver_time_c, aver_nodes, aver_cycles))
    # result.close()



