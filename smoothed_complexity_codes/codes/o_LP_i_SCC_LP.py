import io
import os
import prefpy_io
import math
import time
from numpy import *
import itertools
from preference import Preference
from profile import Profile
import copy
import sys
import networkx as nx
#import rpconfig
from collections import defaultdict
import matplotlib.pyplot as plt
from queue import PriorityQueue
import glob

class MechanismRankedPairs_AAAI_original():
    """
    The Ranked Pairs mechanism.
    """

    def output_graph(self, G):
        # pos = nx.shell_layout(G)  # positions for all nodes
        # print("ok")
        pos = nx.circular_layout(G)  # positions for all nodes
        pos = dict(zip(sorted(pos.keys()), pos.values()))

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=350)

        # edges
        # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
        nx.draw_networkx_edges(G, pos, width=3, alpha=0.5, edge_color='b')

        # labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')

        plt.axis('off')
        plt.savefig("weighted_graph.png")  # save as png
        plt.show()  # display

    def outer_loop_lp(self, profile, prediction=[]):
        """
        Returns a list that associates all the winners of a profile under ranked pairs rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # debug_mode = 0: no output
        # = 1: outputs only initial state
        # = 2: outputs on stop conditions
        # = 3: outputs all data
        BEGIN = time.perf_counter()
        debug_mode = 0

        hits = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        hashtime = 0
        """
        Stopping Condition:
        1: wmg2 U G is acyclic
        2: max_deg == len(I) - 1
        3: possible_winners <= known_winners
        4: len(list_in_deg_0) == len(I) - 1
        5: G U Tier is acyclic
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "soi" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        wmg = profile.getWmg()
        # print(wmg)
        # -------------Initialize the dag G-------------------------
        known_winners = set()
        # ┌---------------- Dec 15 2017 ------------------------┐
        discover = dict()
        TIME = dict()
        # └-----------------------------------------------------┘
        I = list(wmg.keys())
        G = nx.DiGraph()
        wmg2 = nx.DiGraph()
        # ┌---------------- Dec 25 2017 ------------------------┐
        wmg2.add_nodes_from(I)  # found by testing M10N10-20296.csv in m10n10-100k
        # wmg_out_deg = dict(wmg2.out_degree(I))
        # └-----------------------------------------------------┘

        for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
            if wmg[cand1][cand2] > 0:
                wmg2.add_edge(cand1, cand2, weight=wmg[cand1][cand2])

        if debug_mode >= 1:
            print(wmg)
            self.output_graph(wmg2)

        # ┌---------------- Sep 26 2017 ------------------------┐
        num_nodes = 0
        # └-----------------------------------------------------┘
        # ┌---------------- Oct 10 2017 ------------------------
        if nx.is_directed_acyclic_graph(wmg2) is True:
            G_in_degree = wmg2.in_degree(I)
            for i in I:
                if G_in_degree[i] == 0:
                    known_winners.add(i)
                    discover[i] = num_nodes
                    TIME[i] = time.perf_counter() - BEGIN
            return sorted(known_winners), num_nodes, discover, max(discover.values()), TIME, max(TIME.values()), hits, sum(list(hits.values())), hashtime
        # └-----------------------------------------------------------------┘
        root = Node(value=(wmg2, G))
        stackNode = []
        stackNode.append(root)
        outer_nodes = 0

        while stackNode:
            node = stackNode.pop()
            outer_nodes += 1
            (wmg2, G) = node.value
            flag = 0

            if debug_mode == 3:
                print("Popped new node: ")
                print("   wmg2: ", wmg2.edges)
                print("   G: ", G.edges)

            while len(wmg2.edges()) != 0:

                if debug_mode == 3:
                    print("     At top of inner while loop: ")
                    print("        wmg2=", wmg2.edges())

                (max_weight, max_edge) = max([(d['weight'], (u, v)) for (u, v, d) in wmg2.edges(data=True)])
                ties = [d['weight'] for (u, v, d) in wmg2.edges(data=True)].count(max_weight)

                if ties == 1:
                    if debug_mode == 3:
                        print("only 1 tie")

                    G.add_edges_from([max_edge])
                    if nx.is_directed_acyclic_graph(G) is True:
                        wmg2.remove_edges_from([max_edge])
                        # When G is updated, check it immediately
                        # ┌---------------stop condition 1------------------┐
                        temp_G = nx.compose(wmg2, G)
                        if nx.is_directed_acyclic_graph(temp_G) is True:
                            hits[1] += 1
                            G_in_degree = temp_G.in_degree(I)
                            to_be_added = set([x[0] for x in G_in_degree if x[1] == 0])
                            # known_winners = known_winners.union(to_be_added)
                            for c in to_be_added:
                                if c not in known_winners:
                                    known_winners.add(c)
                                    discover[c] = num_nodes
                                    TIME[c] = time.perf_counter() - BEGIN

                            # print(8,set([x[0] for x in G_in_degree if x[1] == 0]))
                            flag = 1
                            if debug_mode >= 2:
                                print("Stop: acyclic - 1 tie")
                            break
                        # ├---------------stop condition 2------------------┤
                        out_deg = G.out_degree(I)
                        (max_cand, max_deg) = max(out_deg, key=lambda x: x[1])
                        if max_deg == len(I) - 1:
                            hits[2] += 1
                            if max_cand not in known_winners:
                                known_winners.add(max_cand)
                                discover[max_cand] = num_nodes
                                TIME[max_cand] = time.perf_counter() - BEGIN
                            # print(7,max_cand)
                            flag = 1
                            if debug_mode >= 2:
                                print("Stop: max out degree - 1 tie")
                            break
                        # ├---------------stop condition 3.1------------------┤
                        in_deg = G.in_degree(I)
                        list_in_deg_0 = [x[0] for x in in_deg if x[1] > 0]
                        possible_winners = set(I) - set(list_in_deg_0)
                        if possible_winners <= known_winners:
                            hits[3] += 1
                            flag = 1
                            break
                        if len(list_in_deg_0) == len(I) - 1:
                            hits[4] += 1
                            to_be_added = [x[0] for x in in_deg if x[1] == 0][0]

                            if to_be_added not in known_winners:
                                known_winners.add(to_be_added)
                                discover[to_be_added] = num_nodes
                                TIME[to_be_added] = time.perf_counter() - BEGIN
                            # print(6,[x[0] for x in in_deg if x[1] == 0][0])
                            if debug_mode >= 2:
                                print("Stop: in degree > 0 - 1 tie")
                            flag = 1
                            break
                        # └---------------------------------------------------┘
                    else:
                        G.remove_edges_from([max_edge])
                        wmg2.remove_edges_from([max_edge])
                    continue
                else:
                    # ties > 1, there are multiple edges with same max weight.
                    edges = [(u, v) for (u, v, d) in wmg2.edges(data=True) if d['weight'] == max_weight]
                    if debug_mode == 3:
                        print("Tied edges =", edges)
                    # Gc = G.copy()
                    # Gc.add_edges_from(edges)
                    wmg2.remove_edges_from(edges)

                    # G, edges, known_winners, flag, discover = self.optimize_tier2(G, edges, known_winners, I, discover, num_nodes)
                    # G, edges, flag = self.optimize_tier3(G, edges)

                    # Compute "inconsistent edges"
                    reverse_G = nx.DiGraph.reverse(G)
                    tier = list(set(edges) - set(nx.transitive_closure(reverse_G).edges()))

                    # edges = new_edges.copy()
                    Gc = G.copy()
                    Gc.add_edges_from(tier)
                    if nx.is_directed_acyclic_graph(Gc) is True:
                        if len(wmg2.edges()) == 0:
                            # self.output_graph(G)
                            Gc_in_degree = Gc.in_degree(I)
                            to_be_added = set([x[0] for x in Gc_in_degree if x[1] == 0])
                            # known_winners = known_winners.union(to_be_added)
                            for c in to_be_added:
                                if c not in known_winners:
                                    known_winners.add(c)
                                    discover[c] = num_nodes
                                    TIME[c] = time.perf_counter() - BEGIN
                            # print(3,set([x[0] for x in G_in_degree if x[1] == 0]))
                        else:
                            G = Gc.copy()
                        continue

                    # Compute "bridge edges" which are not in any cycle
                    # scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if len(g.edges()) != 0]
                    scc = [list(Gc.subgraph(g).edges) for g in nx.strongly_connected_components(Gc) if
                           len(Gc.subgraph(g).edges) != 0]
                    bridges = set(Gc.edges()) - set(itertools.chain(*scc))
                    G.add_edges_from(bridges)
                    tier = list(set(tier) - bridges)
                    # When G is updated, check it immediately
                    # ┌---------------stop condition 2------------------┐
                    out_deg = G.out_degree(I)
                    (max_cand, max_deg) = max(out_deg, key=lambda x: x[1])
                    if max_deg == len(I) - 1:
                        hits[2] += 1
                        if max_cand not in known_winners:
                            known_winners.add(max_cand)
                            discover[max_cand] = num_nodes
                            TIME[max_cand] = time.perf_counter() - BEGIN
                        # print(3, max_cand)
                        if debug_mode >= 2:
                            print("Stop: max out degree - 1 tie")
                        flag = 1
                        break
                    # ├---------------stop condition 3.1------------------┤
                    in_deg = G.in_degree(I)
                    list_in_deg_0 = [x[0] for x in in_deg if x[1] > 0]
                    possible_winners = set(I) - set(list_in_deg_0)
                    if possible_winners <= known_winners:
                        hits[3] += 1
                        flag = 1
                        break
                    if len(list_in_deg_0) == len(I) - 1:
                        hits[4] += 1
                        to_be_added = [x[0] for x in in_deg if x[1] == 0][0]

                        if to_be_added not in known_winners:
                            known_winners.add(to_be_added)
                            discover[to_be_added] = num_nodes
                            TIME[to_be_added] = time.perf_counter() - BEGIN
                        # print(4, [x[0] for x in in_deg if x[1] == 0][0])
                        if debug_mode >= 2:
                            print("Stop: in degree > 0 - 1 tie")
                        flag = 1
                        break
                    # ├---------------continue condition------------------┤
                    #  if tier is currently empty
                    # (i.e. all the edges from last tier are bridge edges)
                    if len(tier) == 0:
                        continue

                    max_children, new_winners, discover, num_nodes, TIME, hits, hashtime = self.scc_based(G, tier, scc, bridges, I, known_winners, discover, num_nodes, TIME, BEGIN, hits, hashtime)
                    known_winners = known_winners.union(new_winners)

                    children = dict()
                    for child in max_children:
                        child_node = Node(value=(wmg2, child))
                        c_in_deg = child.in_degree(I)
                        available = set([x[0] for x in c_in_deg if x[1] == 0])
                        # priority = sum([prediction[y] for y in (available - known_winners)])
                        # ----------------Non-ML Version of local priority Jan 16-------------------
                        priority = len(available - known_winners)
                        children[child_node] = priority
                        continue

                    children_items = sorted(children.items(), key=lambda x: x[1])
                    sorted_children = [key for key, value in children_items]
                    stackNode += sorted_children
                    flag = 1
                    break

            if flag == 0 and len(wmg2.edges()) == 0:
                in_deg = G.in_degree(I)
                to_be_added = set([x[0] for x in in_deg if x[1] == 0])
                # known_winners = known_winners.union(to_be_added)
                for c in to_be_added:
                    if c not in known_winners:
                        known_winners.add(c)
                        discover[c] = num_nodes
                        TIME[c] = time.perf_counter() - BEGIN
                # print("here ", set([x[0] for x in in_deg if x[1] == 0]))
                # print(wmg2.edges())
                # self.output_graph(G)
        # print("hits=%r" % hits)
        return sorted(known_winners), num_nodes, discover, max(discover.values()), TIME, max(TIME.values()), hits, sum(list(hits.values())), hashtime
        # return sorted(known_winners), num_nodes, discover, len_scc



    def edges2string(self, edges, I):
        m = len(I)
        gstring = list(str(0).zfill(m**2))
        for e in edges:
            gstring[(e[0] - min(I))*m + e[1] - min(I)] = '1'

        return ''.join(gstring)

    def string2edges(self, gstring, I):
        m = len(I)
        edges = []
        for i in range(len(gstring)):
            if gstring[i] == '1':
                e1 = i % m + min(I)
                e0 = int((i - e1) / m) + min(I)
                edges.append((e0, e1))
        return edges

    def scc_based(self, G, tier, scc, bridges, I, known_winners, discover, num_nodes, TIME, BEGIN, hits, hashtime):

        if len(scc) == 1:
            children, known_winners_c, discover, num_nodes, TIME, hits, hashtime = self.explore_max_children_lp(G, tier, I, known_winners, discover, num_nodes, TIME, BEGIN, hits, hashtime)
            return children, known_winners_c, discover, num_nodes, TIME, hits, hashtime

        mc_list = []
        # print("ok")

        for x in scc:
            G_temp = nx.DiGraph(list(set(G.edges()).intersection(set(x))))
            T_temp = list(set(tier).intersection(set(x)))
            temp, num_nodes, hits, hashtime = self.explore_max_children_fastest_based_on_scc2(G_temp, T_temp, I, known_winners, 1, 8, num_nodes, hits, hashtime)
            # known_winners = known_winners.union(known_winners_c)
            mc_list.append(temp)
            # mc_list += temp

        Cartesian = itertools.product(*mc_list)
        return [nx.DiGraph(list(set(itertools.chain(*[list(y.edges()) for y in x])).union(bridges))) for x in Cartesian], known_winners, discover, num_nodes, TIME, hits, hashtime

    def explore_max_children_lp(self, G, edges, I, known_winners, discover, num_nodes, TIME, BEGIN, hits, hashtime):
        """
        Problematic
        :param G: DiGraph, A directed graph
        :param edges: list of tuples which correspond to multiple edges with same max weight.
                    e.g. edges = [x for x in wmg2.keys() if wmg2[x] == max_weight]
        :return: set of graphs which correspond to maximum children of given parent: G
        """

        known_winners_c = known_winners.copy()

        # print("hard case")

        # debug_mode = 0: no debug printing
        # debug_mode = 1: prints initial state
        # debug_mode = 2: includes stop conditions
        # debug_mode = 3: most detailed
        debug_mode = 0

        # self.output_graph(G)
        # self.output_graph(nx.DiGraph(edges))

        result = []

        cstack = []

        hashtable = set()
        hashtable2 = set()
        # inhash = 0

        if debug_mode >= 1:
            print("Exploring max children")
            print("G: ", G.edges())
            print("edges: ", edges)

            print("winners:")

            for winner in known_winners:
                print(winner)

            print("-------------")
            print("-------------")

        # ------------- Start Changed Region -----------------------------------------#
        in_deg = G.in_degree()
        nodes_with_no_incoming = set()
        for x in in_deg:
            if x[1] == 0:
                nodes_with_no_incoming.add(x[0])
        for x in I:
            if x not in G.nodes():
                nodes_with_no_incoming.add(x)

        out_deg = G.out_degree()
        (max_cand, max_deg) = max(out_deg, key=lambda x: x[1])

        root = Node(value=(self.edges2string(G.edges(), I), self.edges2string(edges, I), nodes_with_no_incoming,
                           (max_cand, max_deg)))
        # ------------- End Changed Region -----------------------------------------#

        cstack.append(root)

        while cstack:
            node = cstack.pop()

            (G_str, edges_str, no_incoming_, (max_cand_, max_deg_)) = node.value
            num_nodes += 1

            G_ = nx.DiGraph(self.string2edges(G_str, I))
            edges_ = self.string2edges(edges_str, I)
            hash_G_ = hash(G_str)

            if debug_mode == 3:
                print("popped")
                print("G: ", G_.edges())
                print("edges: ", edges_)
                print("-----------")

            if hash_G_ in hashtable2:
                hashtime += 1
                if debug_mode >= 2:
                    print("hashed in hashtable2")
                continue

            # goal state 4: if current G's possible winners is subset of known winners,
            # then directly ignore it.
            if no_incoming_ <= known_winners_c:
                hits[3] += 1
                # ------------------------------ End Changed Region --------------------------------------------#
                if debug_mode >= 2:
                    print("goal state 4 - pruned")
                continue

            # goal state 1: if there are no edges to be added, then add the G_
            if len(edges_) == 0:
                result.append(G_)
                hashtable2.add(hash_G_)

                if debug_mode >= 2:
                    print("goal state 1")
                    print("max child: ", G_.edges())
                continue

            # goal state 2: if union of current G and edges is acyclic,
            # then directly add it to the max_children_set
            G_.add_edges_from(edges_)
            if nx.is_directed_acyclic_graph(G_) is True:
                hits[5] += 1
                hash_temp_G = hash(self.edges2string(G_.edges(), I))
                if hash_temp_G not in hashtable2:
                    G_c = G_.copy()
                    hashtable2.add(hash_temp_G)
                    result.append(G_c)

                    if debug_mode >= 2:
                        print("goal state 2")
                        print("max child: ", G_c.edges())
                continue

            G_.remove_edges_from(edges_)

            # goal state 3: if current G has a candidate whose out-deg reaches m-1,
            # then directly add this candidate to the co-winners set
            if max_deg_ == len(I) - 1:
                hits[2] += 1
                if max_cand_ not in known_winners_c:
                    known_winners_c.add(max_cand_)
                    discover[max_cand_] = num_nodes
                    TIME[max_cand_] = time.perf_counter() - BEGIN
                if debug_mode >= 1:
                    print("adding to known winners in gs3: ", max_cand)
                continue

            # goal state 5: if current G has m-1 candidates whose in-deg > 0,
            # then directly add the remaining one to the co-winners set.
            if len(no_incoming_) == 1:
                hits[4] += 1
                to_be_added = no_incoming_.pop()
                if to_be_added not in known_winners_c:
                    known_winners_c.add(to_be_added)
                    discover[to_be_added] = num_nodes
                    TIME[to_be_added] = time.perf_counter() - BEGIN
                    # known_winners_c.add([x[0] for x in in_deg if x[1] == 0][0])
                    #  -------------------------------- End Changed Region --------------------------------------------#
                if debug_mode >= 1:
                    print("adding to known winners in gs5: ", known_winners_c)
                continue

            isAcyclic = 0
            children = dict()
            index = 0
            for e in edges_:
                if not nx.has_path(G_, source=e[1], target=e[0]):
                    G_.add_edges_from([e])
                    isAcyclic = 1
                    hash_G_c = hash(self.edges2string(G_.edges(), I))
                    if hash_G_c not in hashtable:
                        hashtable.add(hash_G_c)

                        G_c = G_.copy()

                        edge_c = copy.deepcopy(edges_)
                        edge_c.remove(e)

                        # Remove the head of the edge if it had no incoming edges previously
                        no_incoming_c = no_incoming_.copy()
                        no_incoming_c.discard(e[1])

                        max_deg_c = max_deg_
                        max_cand_c = max_cand_
                        if G_c.out_degree(e[0]) > max_deg_:
                            max_deg_c = G_c.out_degree(e[0])
                            max_cand_c = e[0]

                        child = Node(value=(self.edges2string(G_c.edges(), I), self.edges2string(edge_c, I),
                                            no_incoming_c, (max_cand_c, max_deg_c)))

                        priority = len(no_incoming_c - known_winners_c)

                        children[child] = (priority, index)
                        index = index + 1

                        if debug_mode == 3:
                            print("add new child with edge ", e, " and priority ", priority)
                    else:
                        hashtime += 1
                    G_.remove_edges_from([e])

            children_items = sorted(children.items(), key=lambda x: (x[1][0], x[1][1]))
            sorted_children = [key for key, value in children_items]
            cstack += sorted_children

            # goal state 6: if there is no way to add edges
            if isAcyclic == 0:
                hash_G_ = hash(self.edges2string(G_.edges(), I))
                if hash_G_ not in hashtable2:
                    result.append(G_)
                    hashtable2.add(hash_G_)

                    if debug_mode >= 2:
                        print("goal state 6")
                        print("max child: ", G_.edges())
                else:
                    hashtime += 1
                continue

        if debug_mode >= 1:
            print("finished exploring max children")
            print("new winners: ", known_winners_c)

        return result, known_winners_c, discover, num_nodes, TIME, hits, hashtime

    def explore_max_children_fastest_based_on_scc2(self, G, edges, I, known_winners, w_1, w_2, num_nodes, hits, hashtime):
        """
        :param G: DiGraph, A directed graph
        :param edges: list of tuples which correspond to multiple edges with same max weight.
                    e.g. edges = [x for x in wmg2.keys() if wmg2[x] == max_weight]
        :return: set of graphs which correspond to maximum children of given parent: G
        """

        # known_winners_c = known_winners.copy()
        # print("known-winner=", known_winners)
        G.add_nodes_from(I)

        # print("hard case")

        # debug_mode = 0: no debug printing
        # debug_mode = 1: prints initial state
        # debug_mode = 2: includes stop conditions
        # debug_mode = 3: most detailed
        debug_mode = 0

        # self.output_graph(G)
        # self.output_graph(nx.DiGraph(edges))

        result = []

        cpriority = PriorityQueue()

        hashtable = set()
        hashtable2 = set()
        # NODE = 0

        if debug_mode >= 1:
            print("Exploring max children")
            print("G: ", G.edges())
            print("edges: ", edges)

            print("winners:")

            for winner in known_winners:
                print(winner)

            print("-------------")
            print("-------------")

        # ------------- Start Changed Region -----------------------------------------#
        in_deg = G.in_degree()
        nodes_with_no_incoming = set()
        for x in in_deg:
            if x[1] == 0:
                nodes_with_no_incoming.add(x[0])
        for x in I:
            if x not in G.nodes():
                nodes_with_no_incoming.add(x)

        out_deg = G.out_degree()
        (max_cand, max_deg) = max(out_deg, key=lambda x: x[1])

        root = Node(value=(self.edges2string(G.edges(), I), self.edges2string(edges, I), nodes_with_no_incoming,
                           (max_cand, max_deg)))
        # ------------- End Changed Region -----------------------------------------#

        cpriority.put((0, root))

        while not cpriority.empty():

            node_priority = cpriority.get()
            node = node_priority[1]

            (G_str, edges_str, no_incoming_, (max_cand_, max_deg_)) = node.value
            num_nodes += 1

            G_ = nx.DiGraph(self.string2edges(G_str, I))
            edges_ = self.string2edges(edges_str, I)
            hash_G_ = hash(G_str)

            if debug_mode == 3:
                print("popped")
                print("G: ", G_.edges())
                print("edges: ", edges_)
                print("-----------")

            if hash_G_ in hashtable2:
                hashtime += 1
                if debug_mode >= 2:
                    print("hashed in hashtable2")
                continue

            # goal state 4: if current G's possible winners is subset of known winners,
            # then directly ignore it.

            # ------------------------------ Start Changed Region --------------------------------------------#
            # in_deg = G_.in_degree(I)
            # list_in_deg_0 = [x[0] for x in in_deg if x[1] > 0]
            # possible_winners = set(I) - set(list_in_deg_0)
            # if possible_winners <= known_winners_c:
            # if no_incoming_ <= known_winners_c:
            #     # ------------------------------ End Changed Region --------------------------------------------#
            #     if debug_mode >= 2:
            #         print("goal state 4 - pruned")
            #     continue

            # goal state 1: if there are no edges to be added, then add the G_
            if len(edges_) == 0:
                result.append(G_)
                hashtable2.add(hash_G_)

                if debug_mode >= 2:
                    print("goal state 1")
                    print("max child: ", G_.edges())
                continue

            # goal state 2: if union of current G and edges is acyclic,
            # then directly add it to the max_children_set

            #  -------------------------------- Start Changed Region --------------------------------------------#
            # temp_G = G_.copy()
            # temp_G.add_edges_from(edges_)

            G_.add_edges_from(edges_)

            #  -------------------------------- End Changed Region --------------------------------------------#

            if nx.is_directed_acyclic_graph(G_) is True:
                hits[5] += 1
                hash_temp_G = hash(self.edges2string(G_.edges(), I))
                if hash_temp_G not in hashtable2:
                    G_c = G_.copy()
                    hashtable2.add(hash_temp_G)
                    result.append(G_c)

                    if debug_mode >= 2:
                        print("goal state 2")
                        print("max child: ", G_c.edges())
                continue

            # -------------------------------- Start Changed Region --------------------------------------------#
            G_.remove_edges_from(edges_)
            #  -------------------------------- End Changed Region --------------------------------------------#

            # goal state 3: if current G has a candidate whose out-deg reaches m-1,
            # then directly add this candidate to the co-winners set

            # -------------------------------- Start Changed Region --------------------------------------------#
            # out_deg = G_.out_degree(I)
            # (max_cand, max_deg) = max(out_deg, key=lambda x: x[1])

            # if max_deg_ == len(I) - 1:
            #     #  -------------------------------- End Changed Region --------------------------------------------#
            #     if max_cand_ not in known_winners_c:
            #         known_winners_c.add(max_cand_)
            #         # discover[max_cand_] = num_nodes
            #     if debug_mode >= 1:
            #         print("adding to known winners in gs3: ", max_cand)
            #     continue

            # goal state 5: if current G has m-1 candidates whose in-deg > 0,
            # then directly add the remaining one to the co-winners set.

            # -------------------------------- Start Changed Region --------------------------------------------#
            # if len(list_in_deg_0) == len(I) - 1:
            # if len(no_incoming_) == 1:
            #     to_be_added = no_incoming_.pop()
            #     if to_be_added not in known_winners_c:
            #         known_winners_c.add(to_be_added)
            #         # discover[to_be_added] = num_nodes
            #     # known_winners_c.add([x[0] for x in in_deg if x[1] == 0][0])
            # #  -------------------------------- End Changed Region --------------------------------------------#
            #     if debug_mode >= 1:
            #         print("adding to known winners in gs5: ", known_winners_c)
            #     continue

            isAcyclic = 0
            for e in edges_:
                # if nx.is_directed_acyclic_graph(G_) is True:
                # If there is a backward path, then adding this edge causes a cycle

                # -------------------------------- Start Changed Region --------------------------------------------#
                # G_c = G_.copy()
                # G_c.add_edges_from([e])
                # if nx.is_directed_acyclic_graph(G_c) is True:

                if not nx.has_path(G_, source=e[1], target=e[0]):
                    # G_c = G_.copy()
                    # G_c.add_edges_from([e])

                    G_.add_edges_from([e])
                #  -------------------------------- End Changed Region --------------------------------------------#
                    isAcyclic = 1
                    hash_G_c = hash(self.edges2string(G_.edges(), I))
                    if hash_G_c not in hashtable:
                        hashtable.add(hash_G_c)

                        G_c = G_.copy()

                        edge_c = copy.deepcopy(edges_)
                        edge_c.remove(e)

                        # -------------------------------- Start Changed Region ------------------------------------#
                        # Remove the head of the edge if it had no incoming edges previously
                        no_incoming_c = no_incoming_.copy()
                        no_incoming_c.discard(e[1])

                        max_deg_c = max_deg_
                        max_cand_c = max_cand_
                        if G_c.out_degree(e[0]) > max_deg_:
                            max_deg_c = G_c.out_degree(e[0])
                            max_cand_c = e[0]

                        child = Node(value=(self.edges2string(G_c.edges(), I), self.edges2string(edge_c, I),
                                            no_incoming_c, (max_cand_c, max_deg_c)))
                        #  -------------------------------- End Changed Region --------------------------------------#

                        # NODE += 1

                        priority = 0
                        if e[0] in G_.nodes():
                            if e[0] not in known_winners:
                                if e[0] in no_incoming_:
                                    priority = - w_1 * G_.out_degree(e[0])
                                else:
                                    priority = float('inf')
                            else:
                                if e[1] in G_.nodes():
                                    if e[1] in no_incoming_ and e[1] not in known_winners:
                                        priority = - w_2 * G_.out_degree(e[1])
                                    else:
                                        priority = float('inf')
                                else:
                                    priority = float('inf')

                        cpriority.put((priority, child))

                        if debug_mode == 3:
                            print("add new child with edge ", e, " and priority ", priority)
                    else:
                        hashtime += 1
                    G_.remove_edges_from([e])

            # goal state 6: if there is no way to add edges
            if isAcyclic == 0:
                hash_G_ = hash(self.edges2string(G_.edges(), I))
                if hash_G_ not in hashtable2:
                    result.append(G_)
                    hashtable2.add(hash_G_)

                    if debug_mode >= 2:
                        print("goal state 6")
                        print("max child: ", G_.edges())
                else:
                    hashtime += 1
                continue

        # print("result=", result)
        return result, num_nodes, hits, hashtime


class Node:
    def __init__(self, value=None):
        self.value = value

    def __lt__(self, other):
        return 0

    def getvalue(self):
        return self.value

def read_Y_result(inputfile):
    Y = dict()
    temp = inputfile.readline()
    filenames = []
    while temp:
        infomation = temp.strip().split(" =")
        filenames.append(infomation[0])
        # print("%s！"% infomation[0])
        x = infomation[1].split()
        # print(x)
        # x = x.split(', ')
        # print(x)
        Y[infomation[0]] = [ int( x ) for x in x if x ]
        # print(Y[infomation[0]])
        temp = inputfile.readline()

    return Y, filenames


def read_Y_prediction(inputfile):
    Y = dict()
    temp = inputfile.readline()
    filenames = []
    while temp:
        infomation = temp.strip().split(":")
        filenames.append(infomation[0])
        # print(infomation[1])
        x = infomation[1].split()
        # print(x)
        # x = x.split(', ')
        # print(x)
        Y[infomation[0]] = [ float( x ) for x in x if x ]
        # print(Y[infomation[0]])
        temp = inputfile.readline()

    return Y, filenames


if __name__ == '__main__':
    def handler(signum, frame):
        raise AssertionError


    #os.chdir()
    # y1_filenames = rpconfig.filename
    # # y1_filenames = '/Users/junwang/Documents/Social Choice/RP_results_data_and_graph/report-10k-hard-cases-m10n10-0102-2.json.nn'
    # inf1 = open(y1_filenames, 'r')
    # # _, filenames = read_Y_distribution(inf1)
    # prediction, filenames = read_Y_prediction(inf1)
    # inf1.close()

    # y2_filenames = '/Users/junwang/Documents/Social Choice/RP_results_data_and_graph/report-10k-hard-cases-m10n10-0102-2.json.nn'
    # inf2 = open(y2_filenames, 'r')
    # # _, filenames = read_Y_distribution(inf1)
    # prediction, filenames = read_Y_prediction(inf1)
    # inf1.close()



    num_profiles = 0
    total_time = 0
    total_node = 0
    total_100time = 0
    total_100node = 0
    total_hits = 0
    total_hash = 0
    filenames = glob.glob('your choice')
    # filenames = ['M10N10-17221.csv']

    for inputfile in filenames:
        # try:
        #     signal.signal(signal.SIGALRM, handler)
        #     signal.alarm(3)
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)

        rec = inputfile.strip()

        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        start = time.perf_counter()
        # print(prediction[inputfile])

        # prediction2 = [int(x in prediction[inputfile]) for x in range(10)]
        # print("pred=", prediction[inputfile])
        rp_cowinner = MechanismRankedPairs_AAAI_original().outer_loop_lp(profile)
        end = time.perf_counter()
        # print("%s\t%r\t%d\t%r\t%d\t%f" % (inputfile, rp_cowinner[0], rp_cowinner[1], rp_cowinner[2], rp_cowinner[3], (end - start)))
        print("%s\t%r\t%d\t%r\t%d\t%f\t%r\t%f\t%r\t%d\t%d" % (inputfile, rp_cowinner[0], rp_cowinner[1], rp_cowinner[2],
                                                  rp_cowinner[3], (end - start), rp_cowinner[4], rp_cowinner[5], rp_cowinner[6], rp_cowinner[7], rp_cowinner[8]))

        num_profiles += 1
        total_time += end - start
        total_node += rp_cowinner[1]
        total_100time += rp_cowinner[5]
        total_100node += rp_cowinner[3]
        total_hits += rp_cowinner[7]
        total_hash += rp_cowinner[8]
        # signal.alarm(0)
        # except AssertionError:
        #     print("timeout")

    ave_time = total_time / num_profiles
    ave_node = total_node / num_profiles
    ave_100time = total_100time / num_profiles
    ave_100node = total_100node / num_profiles
    ave_hits = total_hits / num_profiles
    ave_hash = total_hash / num_profiles
    print("#profiles\n#ave_node\n#ave_100node\nave_time\nave_100time\nave_hits\nave_hash\n%f\n%f\n%f\n%f\n%f\n%f\n%f" % (num_profiles, ave_node, ave_100node, ave_time, ave_100time, ave_hits, ave_hash))