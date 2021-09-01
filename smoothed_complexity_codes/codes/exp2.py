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
# import cycles
import sys
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# import prefpy
# from .preference import Preference
# from .profile import Profile
# from prefpy import preference

import warnings
warnings.filterwarnings("ignore")

# def simple_cycles(G):
#     def _unblock(thisnode):
#         """Recursively unblock and remove nodes from B[thisnode]."""
#         if blocked[thisnode]:
#             blocked[thisnode] = False
#             while B[thisnode]:
#                 _unblock(B[thisnode].pop())
#
#     def circuit(thisnode, startnode, component):
#         closed = False # set to True if elementary path is closed
#         path.append(thisnode)
#         blocked[thisnode] = True
#         for nextnode in sorted(component[thisnode]): # direct successors of thisnode
#             if nextnode == startnode:
#                 result.append(path + [startnode])
#                 closed = True
#             elif not blocked[nextnode]:
#                 if circuit(nextnode, startnode, component):
#                     closed = True
#         if closed:
#             _unblock(thisnode)
#         else:
#             for nextnode in component[thisnode]:
#                 if thisnode not in B[nextnode]: # TODO: use set for speedup?
#                     B[nextnode].append(thisnode)
#         path.pop() # remove thisnode from path
#         return closed
#
#     path = [] # stack of nodes in current path
#     blocked = defaultdict(bool) # vertex: blocked from search?
#     B = defaultdict(list) # graph portions that yield no elementary circuit
#     result = [] # list to accumulate the circuits found
#     # Johnson's algorithm requires some ordering of the nodes.
#     # They might not be sortable so we assign an arbitrary ordering.
#     ordering=dict(zip(sorted(G),range(len(G))))
#     for s in sorted(ordering.keys()):
#         # Build the subgraph induced by s and following nodes in the ordering
#         subgraph = G.subgraph(node for node in G
#                               if ordering[node] >= ordering[s])
#         # Find the strongly connected component in the subgraph
#         # that contains the least node according to the ordering
#         strongcomp = nx.strongly_connected_components(subgraph)
#         mincomp=min(strongcomp,
#                     key=lambda nodes: min(ordering[n] for n in nodes))
#         component = G.subgraph(mincomp)
#         if component:
#             # smallest node in the component according to the ordering
#             startnode = min(component,key=ordering.__getitem__)
#             for node in component:
#                 blocked[node] = False
#                 B[node][:] = []
#             dummy=circuit(startnode, startnode, component)
#
#     return result

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
    n = np.sum(np.abs([wmg[0][i] for i in range(1,m)]))
    wmg_vec = [wmg[i][j] for i in range(m) for j in range(m) if not j == i]
    print("wmg_vec=", wmg_vec)
    wmg_vec_normalized = list(1.*np.array(wmg_vec)/n)
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
        print('G2=',G)
        print(cyclic(G))

        if cyclic(G) is False:
            wmg2.pop(edge)
            print("ok",G)
        else:
            G[edge[0]].remove(edge[1])
            wmg2.pop(edge)
    print('G=',G)
    top_list = topological_sort(G)
    return top_list[0]


def DiGraph_transform(graph):
    # graph = {1: (2,), 2: (3,), 3: (1,)}
    G_list = []
    for key in graph.keys():
        for value in graph[key]:
            G_list.append((key, value))
    return nx.DiGraph(G_list)


if __name__ == '__main__':

    os.chdir('G:\soc-3-hardcase')
    # inputfile = "M10N10-26.csv"

    filenames = glob.glob('M10N10-*.csv')
    filenames = filenames[0:200]


    for inputfile in filenames:
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)

        rec = inputfile.strip()

        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)
        start = time.perf_counter()
        rp_cowinner = MechanismRankedPairs().ranked_pairs_cowinners_wo_cycles(profile)
        # rp_cowinner = MechanismRankedPairs().ranked_pairs_cowinners_DFS(profile)
        end = time.perf_counter()
        print(inputfile, "= ", rp_cowinner, "= %f" % (end - start))

    # T1 = nx.DiGraph([(4, 7), (8, 7), (9, 1), (3, 7)])
    # T2 = nx.DiGraph([(0, 1), (0, 4), (0, 7), (8, 4), (9, 2), (9, 3), (9, 7), (3, 1), (3, 2), (3, 6)])
    # T3 = nx.DiGraph([(0, 8), (0, 9), (1, 7), (4, 1), (4, 2), (8, 2), (8, 6), (9, 4), (9, 6), (9, 8), (2, 0), (2, 1), (2, 6), (2, 7), (6, 1), (6, 7), (3, 5), (5, 6), (5, 7), (5, 8)])

    # MechanismRankedPairs().output_graph(T1)
    # MechanismRankedPairs().output_graph(T2)
    # MechanismRankedPairs().output_graph(T3)

    # G3 = nx.DiGraph([(4, 7), (8, 7), (8, 4), (9, 1), (9, 2), (9, 3), (9, 7), (3, 7), (3, 1), (3, 2), (3, 6), (0, 1), (0, 4), (0, 7)])
    # G3.add_nodes_from([5])
    # MechanismRankedPairs().output_graph(G3)

    # filenames = glob.glob('M20N20-*.csv')
    # filenames = filenames[0:1000]

    # os.chdir('D:\Social Choice\data\soc')
    # inputfile = "ED-00004-00000025.soc"

    # os.chdir('D:\Social Choice\data\soc-toc-jw')
    # inputfile = "ED-10005-00000004.soc"

    # os.chdir('D:\Social Choice\data\soc-4')
    # inputfile = "M10N10-1.csv"

    # os.chdir('D:\Social Choice\data')
    # inputfile = "stv-m20n20-7000/M20N20-5198.csv"

    # inf = open(inputfile, 'r')
    # cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
    #
    # rec = inputfile.strip()
    #
    # inf.close()
    # profile = Profile(cmap, preferences=[])
    # Profile.importPreflibFile(profile, inputfile)

    # G = {0: {1}, 1: {2}, 2: {3}, 3: set(), 4: set()}
    # G = [(0, 1), (1, 2), (2, 3)]
    # G = []
    # I = list(range(3))
    # result = MechanismRankedPairs().DiGraph_inverse_transform(G, I)
    # print("res=", result)
    # edges = [(0, 2), (0, 3), (0, 4), (1, 3), (2, 4), (3, 4), (4, 1)]
    # edges= [(0, 8), (1, 8), (2, 3), (4, 1), (5, 3), (6, 1), (6, 2), (6, 3), (8, 2), (8, 5)]
    # edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4,5), (5,6), (6,7),(7,0)]
    # edges = [(3, 9), (9, 6), (6, 8), (5, 2), (1, 0), (8, 1), (6, 5), (7, 8)]
    # edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 1)]
    # G = [(4, 5), (4, 3), (5, 1), (2, 1), (2, 3), (2, 4), (2, 7), (3, 1), (7, 0), (7, 1), (7, 3), (7, 4), (7, 9), (7, 5), (0, 3), (0, 4), (9, 1), (9, 4), (9, 0), (9, 5), (6, 3), (6, 4), (6, 0), (6, 1), (6, 2), (8, 3), (8, 0), (8, 5)]
    # edges=[]
    # G = [(1,2),(2,5),(5,1),(2,6),(5,6),(2,3),(3,4),(4,8),(8,3),(3,7),(8,7),(6,7),(7,6)]

    # G = [(9, 3), (1, 9), (1, 3), (0, 9), (0, 3), (6, 3), (2, 0), (2, 3), (7, 3), (8, 0), (8, 3), (5, 3), (4, 3), (1, 6), (7, 2), (4, 0), (2, 9), (8, 1), (7, 6), (5, 0), (4, 1), (6, 4), (8, 2), (4, 5), (6, 0), (7, 5), (8, 7), (7, 9), (7, 0), (7, 4), (9, 5), (5, 2), (2, 4), (8, 4)]
    # G = [(0, 4), (0, 3), (0, 5), (0, 1), (0, 2), (0, 6), (4, 5), (3, 8), (3, 4), (3, 5), (3, 6), (3, 2), (8, 6), (8, 7), (7, 6), (7, 4), (5, 1), (5, 6), (1, 6), (1, 2), (9, 8), (9, 1), (9, 2), (9, 4), (9, 5), (9, 6), (2, 6)]
    # G = nx.DiGraph(G)

    # E = nx.DiGraph(edges)
    # Gc = G.copy()
    # Gc.add_edges_from(edges)

    # odeg = G.out_degree(I)
    # print("out degree=", odeg)

    # cycles = list(nx.simple_cycles(G))
    # print("cycles=", cycles)
    # scc = nx.strongly_connected_components(G)
    # # print((8,2) in scc)
    # l_scc = list(scc)
    # print(l_scc)
    #
    # print(len(l_scc))
    # print(len(list(scc)))
    # # safe_edges=[]
    # scc2 = nx.strongly_connected_component_subgraphs(G, copy=True)
    # for g in scc2:
    #     print(g.edges())
    # # print(list(scc2)[1].edges())


    # I = range(10)
    # start0 = time.perf_counter()
    # result, known_winners = MechanismRankedPairs().explore_max_children(G, edges, I)
    # end0 = time.perf_counter()
    # print(result, known_winners)

    # for gc in Gc:
    #     print(gc.edges())
    # print("NODE=", NODE, "time=%f s" % (end0 - start0))
    # G = nx.DiGraph(edges)
    # b = nx.is_directed_acyclic_graph(G)
    # print(list(nx.simple_cycles(G)))
    # print("bool=",b)

    # # # # ---------------------------plotting --------------------------------------------------------
    # # pos = nx.shell_layout(G)  # positions for all nodes
    # pos = nx.circular_layout(G)  # positions for all nodes
    # pos = dict(zip(sorted(pos.keys()), pos.values()))
    #
    # # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=350)
    #
    # # edges
    # # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    # nx.draw_networkx_edges(G, pos, width=3, alpha=0.5, edge_color='b')
    # # nx.draw_networkx_edges(E, pos, width=3, alpha=0.5, edge_color='b')
    #
    # # labels
    # nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')
    #
    # plt.axis('off')
    # plt.savefig("weighted_graph.png")  # save as png
    # plt.show()  # display

    #
    # wmg = profile.getWmg()
    # print(wmg)



    # G0 = {1: (2,), 2: (3,), 3: (1,2)}
    # G1 = DiGraph_transform(G0)
    # result1 = MechanismRankedPairs().simple_cycles(G1)
    # print(result1)
    #
    # G = nx.DiGraph([(3, 1), (1, 2), (2, 3), (1, 4), (4, 3)])
    # result = MechanismRankedPairs().simple_cycles(G)
    # print(result)



    # single_winner = MechanismRankedPairs().ranked_pairs_single_winner(profile)
    # print("RANKED PAIR single_winner = ", single_winner)

    # start = time.perf_counter()
    # rp_cowinner = MechanismRankedPairs().ranked_pairs_cowinners_wo_cycles(profile)
    # end = time.perf_counter()
    # print("RANKED PAIR cowinners = ", rp_cowinner, "time = %f s" % (end - start))


    '''
    # black_winner = MechanismBlack().black_winner(profile)
    # print("black_winner = ", black_winner)
    Borda_winner = MechanismBorda().getWinners(profile)
    print("Borda_winner = ", Borda_winner)
    Borda_mov = MechanismBorda().getMov(profile)
    Borda_map = MechanismBorda().getCandScoresMap(profile)
    print("Borda_map = ", Borda_map)
    print("Borda_mov = ", Borda_mov)

    print("-------------------")
    ka_mov = MechanismKApproval(3).getMov(profile)
    ka_winners = MechanismKApproval(3).getWinners(profile)
    ka_scoremap = MechanismKApproval(3).getCandScoresMap(profile)
    print("ka_scoremap=", ka_scoremap)
    print("ka_winners=", ka_winners)
    print("ka-mov=", ka_mov)
    print("-------------------")

    plural_winner = MechanismPlurality().getWinners(profile)
    plu_map = MechanismPlurality().getCandScoresMap(profile)
    print("plu_map", plu_map)
    print("plural_winner = ", plural_winner)
    plural_mov = MechanismPlurality().getMov(profile)
    print("plural_mov = ", plural_mov)

    veto_winner = MechanismVeto().getWinners(profile)
    print("veto_winner = ", veto_winner)
    veto_map = MechanismVeto().getCandScoresMap(profile)
    print("veto_map = ", veto_map)
    veto_mov = MechanismVeto().getMov(profile)
    print("veto_mov = ", veto_mov)

    Bucklin_winner = MechanismSimplifiedBucklin().getWinners(profile)
    print("Bucklin_winner = ", Bucklin_winner)
    Bucklin_mov = MechanismSimplifiedBucklin().getMov(profile)
    print("Bucklin_mov = ", Bucklin_mov)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cowinners = MechanismRankedPairs().ranked_pairs_cowinners(profile)
    print("RANKED PAIR cowinners = ", cowinners)

    black_winner = MechanismBlack().black_winner(profile)
    print("black_winner = ", black_winner)

    pwro_winner = MechanismPluralityRunOff().PluRunOff_single_winner(profile)
    print("pwro_winner = ", pwro_winner)
    pwro_winners = MechanismPluralityRunOff().PluRunOff_cowinners(profile)
    print("pwro_winners = ", pwro_winners)
    pwro_mov = MechanismPluralityRunOff().getMov(profile)
    print("pwro_mov = ", pwro_mov)

    winners_sntv = MechanismSNTV().SNTV_winners(profile, 2)
    mov_sntv = MechanismSNTV().getMov(profile, 2)
    print("winners_sntv=", winners_sntv, "mov_sntv=", mov_sntv)

    plu_scores = MechanismPlurality().getCandScoresMap(profile)
    print(plu_scores)
    '''
    # bm_winners = MechanismBordaMean().Borda_mean_winners(profile)
    # print("bordamean winners=", bm_winners)

    # cc_winners = MechanismChamberlin_Courant().single_peaked_winners(profile, d=1, K=3, funcType='Borda')
    # print("cc_winners=", cc_winners)

    # prefcounts = profile.getPreferenceCounts()
    # rankmaps = list(profile.candMap.keys())
    # print(rankmaps)


    '''
    filenames = glob.glob('*.soc')
    num_profile = 0
    time1 = 0
    time2 = 0

    for inputfile in filenames:

        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)
        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)
        start = time.perf_counter()
        mov1 = MoVPluRunOff_1(profile)
        end1 = time.perf_counter()
        mov2 = MoVPluRunOff_2(profile)
        end2 = time.perf_counter()
        print("%r: mov1=%d, mov2=%d; time1=%f s, time2=%f s" % (inputfile, mov1,  mov2, (end1-start), (end2-end1)))
        num_profile += 1
        time1 += end1-start
        time2 += end2-end1

    aver_time1 = time1 / num_profile
    aver_time2 = time2 / num_profile
    print("aver_time1=", aver_time1, "aver_time2=", aver_time2)
    
    '''

