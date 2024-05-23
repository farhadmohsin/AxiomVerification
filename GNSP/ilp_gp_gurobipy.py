import numpy as np
from voting_utils import *
from bruteforce import *
from mallows_gen import *
from itertools import combinations, permutations
from scipy.optimize import linprog
from cvxopt import glpk
import cvxopt
import copy
from time import time
from ilp_gp import create_new_pref
from datetime import datetime
from tqdm import tqdm
import cvxpy as cp
import gurobipy as gp
from pprint import pprint

def combs(a, r):
    b = np.fromiter(combinations(a, r), np.dtype([('', a.dtype)]*r))
    return b.view(a.dtype).reshape(-1, r)

def prefab(a, b, ranking):
    """
    Returns
    -------
    Whether a \succ b in ranking
    """
    ind_a = np.argwhere(ranking==a).flatten()[0]
    ind_b = np.argwhere(ranking==b).flatten()[0]
        
    if ind_a < ind_b: # ind_a < ind_b means a is ranked higher
        return 1
    return -1

def R_top_a(a, ranking, removed = []):
    """
    Parameters
    ----------
    a : alternative.
    ranking : 
    removed : list of eliminated alternatives.

    Returns
    -------
    whether a = top(rankings - removed)
    """
    
    for j in ranking:
        if j == a:
            return 1
        if j in removed:
            continue
        else:
            return 0
        
def R_rank_a(a, ranking):
    """
    Returns
    -------
    Returns a's score in ranking.
    """
    m = len(ranking)
    rank = np.argwhere(ranking == a).flatten()[0]
    return m - rank - 1

def Rab(a, b, rankings):
    """
    Parameters
    ----------
    a, b : alternatives
    rankings : list of rankings

    Returns
    -------
    indices of rankings that have a \succ b
    """
    
    ab_indices = [i for i in range(len(rankings)) if (prefab(a,b,rankings[i]) > 0)]    
    return ab_indices

def e(m, i):
    """returns [0 ... 1 ... 0] with e[i] = 1 """
    return [1 if j==i else 0 for j in range(m)]

def neg(a):
    return [-x for x in a]
    
#%%

def create_LP_old(anon_votes, b, vbw, vwb, C, rankings):
    A = [] # A : need to look into vbw ^^
    h = [] # h: need to look into vwb
    for j in C:
        # print(f'comparing b={b} and j={j}')
        temp = []
        for k in vbw:
            temp.append(-1 * prefab(b, j, rankings[k]))
        A.append(temp)
        
        h0 = 0
        for k in vwb:
            h0 += int(anon_votes[k][0]) * prefab(b, j, rankings[k])
        h.append(h0)
    for l, k in enumerate(vbw):
        A.append(e(len(vbw),l))
        h.append(int(anon_votes[k][0]))
        A.append(neg(e(len(vbw),l)))
        h.append(0)
    
    # print(A)
    # print(h)
    
    A = cvxopt.matrix(A, tc = 'd')
    h = cvxopt.matrix(h, tc = 'd')
    c = cvxopt.matrix([-1 for k in range(len(vbw))], tc = 'd')

    return A, h, c

#%%
cvxopt.solvers.options['show_progress'] = False

def create_LP_Copeland(anon_votes, vbw, vwb, edges, umg, rankings):
    A = [] # A : need to look into vbw ^^
    h = [] # h: need to look into vwb
    
    for i, v in enumerate(umg):
        e1, e2 = edges[i]
        if v == 0: #tie
            temp = []
            temp2 = []
            for k in vbw:
                temp.append(-1 * prefab(e1, e2, rankings[k]))
                temp2.append(prefab(e1, e2, rankings[k]))
            A.append(temp)
            A.append(temp2)
            
            h0 = 0
            for k in vwb:
                h0 += int(anon_votes[k][0]) * prefab(e1, e2, rankings[k])
            h.append(h0) 
            h.append(-h0)
            
        elif v == 1: # e2 \succ e1
            temp = []
            for k in vbw:
                temp.append(-1 * prefab(e2, e1, rankings[k]))
            A.append(temp)
            
            h0 = 0
            for k in vwb:
                h0 += int(anon_votes[k][0]) * prefab(e2, e1, rankings[k])
            h.append(h0 - 1)
            
        elif v == -1: # e1 \succ e2
            temp = []
            for k in vbw:
                temp.append(-1 * prefab(e1, e2, rankings[k]))
            A.append(temp)
            
            h0 = 0
            for k in vwb:
                h0 += int(anon_votes[k][0]) * prefab(e1, e2, rankings[k])
            h.append(h0 - 1)
            
    for l, k in enumerate(vbw):
        A.append(e(len(vbw),l))
        h.append(int(anon_votes[k][0]))
        A.append(neg(e(len(vbw),l)))
        h.append(0)
    
    #print('A', len(A), len(A[0]), 'h', len(h))
    
    A = cvxopt.matrix(A, tc = 'd')
    h = cvxopt.matrix(h, tc = 'd')
    c = cvxopt.matrix([-1 for k in range(len(vbw))], tc = 'd')
    
    return A, h, c

def ILP_Copeland_lexicographic(votes, anon_votes, n, m, UMGS, edges, debug = False):
    
    # initializtion
    w_, s = Copeland_winner(votes)
    w = lexicographic_tiebreaking(votes, w_)
    
    rankings = [av[1] for av in anon_votes]
    # print(votes)
    
    # building the LP
    
    # w is the winner
    # say b is the target winner
    
    k_min = np.inf
    x_min = None
    lp_flag = 0
    
    # go through all possible umgs
    for g, umg in (enumerate(UMGS)):
        #print(f'UMG: {g}')
        b, _ = Copeland_winner_from_umg(umg, m, edges)
        if b == w:
            continue
        
        # print('b', b, 'umg', umg)
        vbw = Rab(b, w, rankings)   # ranking indices with b \succ w
        vwb = Rab(w, b, rankings)   # ranking indices with w \succ b
        
        len_x = len(vbw) # no. of variables
        if(len(vbw) == 0):
            continue
        
        # check to see if b can be possible winner
        
        # form an LP
        A, h, c = create_LP_Copeland(anon_votes, vbw, vwb, edges, umg, rankings)
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.lp(c, A.T, h, options={'show_progress': False})
        # print(sum(c.T * sol['x']))
        if(sol['status'] != 'optimal'):
            continue
        lp_flag = 1
        
        (status, x) = cvxopt.glpk.ilp(c, A.T, h, I=set(range(len(c))))
        if(status != 'optimal'):
            if(debug):
                print('ILP not feasible', 'status')
            continue
        else:
            if(debug):
                print('ILP feasible')
            # c.T*x is the number of voters who has voted, not who's abstained
            # so to get number of abstanation, we can do the following
            k = sum([anon_votes[vv][0] for vv in vbw]) + sum(c.T * x)
            # print('k:', k)
            
            if k < k_min:
                k_min = k
                # x_min = x
                # print('rankings:', [rankings[vv] for vv in vbw])
                # print('x:', x)
                
                new_votes = create_full_pref(create_new_pref(anon_votes, vbw, list(x)))
                votes_min = new_votes
                
                new_w_, new_s = Copeland_winner(new_votes)
                new_w = lexicographic_tiebreaking(votes, new_w_)
                # print('found something', 'w', w, 'b', b, umg)
                # print(len(votes), len(new_votes))
                # print((votes))
                # print((new_votes))
                # print('original scores:', s)
                # print('new_scores:', new_s)
    
    if k_min == np.inf:
        return 0, None, None, lp_flag
    return 1, k_min, votes_min, lp_flag

def ILP_Copeland_lexicographic_new(votes, anon_votes, n, m, debug = False):
    a_, s = Copeland_winner(votes)
    a = lexicographic_tiebreaking(votes, a_)
    
    rankings = [av[1] for av in anon_votes]
    
    k_min = np.inf
    lp_flag = 0

    for b in range(m):
        if b == a:
            continue

        vba = Rab(b, a, rankings)   # ranking indices with b \succ a
        vab = Rab(a, b, rankings)   # ranking indices with a \succ b

        if(len(vba) == 0):
            continue

        model = gp.Model("copeland")
        q = dict()
        r = dict()
        x = dict()
        P_x = dict()
        score = dict()

        for i, R_i in enumerate(vba):
            x[i] = model.addVar(vtype=gp.GRB.INTEGER, name=f"x_{i}")
            model.addConstr(x[i] >= 0)
            model.addConstr(x[i] <= int(anon_votes[R_i][0]))

        for c in range(m):
            for d in range(c + 1, m):
                q[(c, d)] = model.addVar(vtype=gp.GRB.BINARY, name=f"q_{c}_{d}")
                q[(d, c)] = model.addVar(vtype=gp.GRB.BINARY, name=f"q_{d}_{c}")
                r[(c, d)] = model.addVar(vtype=gp.GRB.BINARY, name=f"r_{c}_{d}")
        
                model.addConstr(q[(c, d)] + r[(c, d)] + q[(d, c)] == 1)
        
                P_x[(c, d)] = 0
                P_x[(d, c)] = 0
                for i, R_i in enumerate(vba):
                    P_x[(c, d)] += x[i] * (prefab(c, d, rankings[R_i]) > 0)
                    P_x[(d, c)] += x[i] * (prefab(d, c, rankings[R_i]) > 0)
                for i, R_i in enumerate(vab):
                    P_x[(c, d)] += int(anon_votes[R_i][0]) * (prefab(c, d, rankings[R_i]) > 0)
                    P_x[(d, c)] += int(anon_votes[R_i][0]) * (prefab(d, c, rankings[R_i]) > 0)

                model.addConstr((q[(c, d)] == 1) >> (P_x[(c, d)] >= P_x[(d, c)] + 1))
                model.addConstr((q[(d, c)] == 1) >> (P_x[(d, c)] >= P_x[(c, d)] + 1))
                model.addConstr((r[(c, d)] == 1) >> (P_x[(c, d)] == P_x[(d, c)]))

        for c in range(m):
            score[c] = 0
            for d in range(m):
                if c == d:
                    continue
                # replace 0.5 by alpha
                score[c] += q[(c, d)] + 0.5 * r[(c, d) if c < d else (d, c)]
        
        # ensure b is the winner
        # lexicographic tie-breaking
        for c in range(m):
            if b == c:
                continue
            elif b > c:
                model.addConstr(score[b] >= score[c] + 1e-3) # score[b] > score[c] but gurobi doesn't support strict inequality
            else:
                model.addConstr(score[b] >= score[c])

        model.setObjective(-gp.quicksum(x[i] for i in range(len(vba))), gp.GRB.MINIMIZE)

        model.optimize()
        if debug:
            print(model.status)
        if model.status != gp.GRB.OPTIMAL:
            continue
        lp_flag = 1

        k = sum([anon_votes[vv][0] for vv in vba]) + model.ObjVal

        if debug:
            print('=' * 10)
            print(f'original winner = {a}, possible winner = {b}')
            print(f'k = {k}')
            print(anon_votes)
            for v in model.getVars():
                print(v.varName, v.x)
            print('=' * 10)

        if k < k_min:
            k_min = k
    
    if k_min == np.inf:
        return 0, None, None, lp_flag
    return 1, k_min, None, lp_flag



def create_LP_maximin(anon_votes, b, vbw, vwb, pair, rankings):
    A = [] # A : need to look into vbw ^^
    h = [] # h: need to look into vwb
    
    # print(f'\tStart Printing, winner is {b}')
    # print(f'{pair=}')
    m = len(rankings[0])
    for i, j in enumerate(pair):
        # maximin(i) = #i \succ j
        for l in range(m):
            if l==i or l==j:
                continue
            # print(f'{i}>{j} < {i}>{l}')
            #{i} \succ {j} <= {i} \succ {l}
            
            temp = []
            for k in vbw:
                # update A
                temp.append(prefab(i, j, rankings[k]) - prefab(i, l, rankings[k]))
            A.append(temp)
            
            h0 = 0
            for k in vwb:
                # update h
                h0 += int(anon_votes[k][0]) * (prefab(i, l, rankings[k]) - 
                                               prefab(i, j, rankings[k]))
            h.append(h0)
        
    # b is the winner
    # print('\tstart')
    for i, j in enumerate(pair):
        if i == b:
            continue
        # print(f'{i}>{j} <= {b}>{pair[b]}')
        if b < i:
            #f'{b} \succ {pair[b]} >= {i} \succ {j}'
            temp = []
            for k in vbw:
                # update A
                temp.append(prefab(i, j, rankings[k]) - prefab(b, pair[b], rankings[k]))
            A.append(temp)
            
            h0 = 0
            for k in vwb:
                # update h
                h0 += int(anon_votes[k][0]) * (prefab(b, pair[b], rankings[k]) - 
                                               prefab(i, j, rankings[k]))
            h.append(h0)
        else:
            #f'{b} \succ {pair[b]} > {i} \succ {j}'
            temp = []
            for k in vbw:
                # update A
                temp.append(prefab(i, j, rankings[k]) - prefab(b, pair[b], rankings[k]))
            A.append(temp)
            
            h0 = 0
            for k in vwb:
                # update h
                h0 += int(anon_votes[k][0]) * (prefab(b, pair[b], rankings[k]) - 
                                               prefab(i, j, rankings[k]))
            h.append(h0 - 1)
    # print('\tEnd Printing')            
    for l, k in enumerate(vbw):
        A.append(e(len(vbw),l))
        h.append(int(anon_votes[k][0]))
        A.append(neg(e(len(vbw),l)))
        h.append(0)
    
    # print('A', len(A), len(A[0]), 'h', len(h))
    
    # for cvxopt
    # A = cvxopt.matrix(A, tc = 'd')
    # h = cvxopt.matrix(h, tc = 'd')
    # c = cvxopt.matrix([-1 for k in range(len(vbw))], tc = 'd')
    
    # for cvxpy
    A = np.array(A)
    h = np.array(h)
    c = np.array([-1 for k in range(len(vbw))])
    
    # print(A)
    # print(h)
    # print(c)
    
    return A, h, c


def ILP_maximin_lexicographic(votes, anon_votes, n, m, maximin_pairs, debug = False):
    # initializtion
    w_, s = maximin_winner(votes)
    w = lexicographic_tiebreaking(votes, w_)
    
    print(f'\tmaximin_winner:{w_=}, {s=}')
    
    rankings = [av[1] for av in anon_votes]
    # print(votes)
    
    # building the LP
    
    # w is the winner
    # say b is the target winner
    
    k_min = np.inf
    x_min = None
    lp_flag = 0
    
    for b in range(m):
        if(debug):
            print(f'\tNow checking {b=}')
        # for each winner, we need to go through all possible maximin_pairs
        if b == w:
            continue
        
        vbw = Rab(b, w, rankings)   # ranking indices with b \succ w
        vwb = Rab(w, b, rankings)   # ranking indices with w \succ b
        
        len_x = len(vbw) # no. of variables
        if(debug):
            print(f'{len(rankings)=}, {len_x=}')
        if(len(vbw) == 0):
            continue
        
        for p, pair in (enumerate(maximin_pairs)):
            # form an LP
            #print(f'\t\t{pair}')
            
            
            A, h, c = create_LP_maximin(anon_votes, b, vbw, vwb, pair, rankings)
            
            # for cvxpy
            
            # construct the problem
            x0 = cp.Variable(len_x)
            # constraint = A @ x <= h
            # objective = cp.minimize(c @ x)
            
            # creating separate versions so that status etc. might be addressed
            problem_LP = cp.Problem(cp.Minimize(c @ x0), [A @ x0 <= h])
            
            # solve LP
            problem_LP.solve()
            if problem_LP.status not in ["infeasible", "unbounded"]:
                lp_flag = 1
                if(debug):
                    print(pair, 'LP feasible')
                    #print('\t', [anon_votes[vv][0] for vv in vbw])
            else:
                continue
            
            # solve ILP
            x1 = cp.Variable(len_x, integer = True)
            problem = cp.Problem(cp.Minimize(c @ x1), [A @ x1 <= h])
            problem.solve(solver='GUROBI')
            
            if problem.status in ["infeasible", "unbounded"]:
                if(debug):
                    print('ILP not feasible', 'status')
                continue
            else:
                k = sum([anon_votes[vv][0] for vv in vbw]) + c @ x1.value
                if(debug):
                    print('ILP feasible', problem.status, x1.value)
                    print('k:', k)
                
                if k < k_min:
                    k_min = k
                    
                new_votes = create_full_pref(create_new_pref(anon_votes, vbw, list(x1.value)))
                votes_min = new_votes
                
                new_w_, new_s = maximin_winner(new_votes)
                new_w = lexicographic_tiebreaking(votes, new_w_)
                print('found something', 'w', w, 'b', b, 'new_w', new_w)
                # print(len(votes), len(new_votes))
                # print((votes))
                # print((new_votes))
                print('original scores:', s)
                print('new_scores:', new_s)
            
    if k_min == np.inf:
        return 0, k_min, lp_flag
    return 1, k_min, lp_flag


def ILP_maximin_lexicographic_new(votes, anon_votes, n, m, debug = False):
    a_, s = maximin_winner(votes)
    a = lexicographic_tiebreaking(votes, a_)
    
    rankings = [av[1] for av in anon_votes]
    
    k_min = np.inf
    lp_flag = 0

    for b in range(m):
        if b == a:
            continue
        
        vba = Rab(b, a, rankings)   # ranking indices with b \succ a
        vab = Rab(a, b, rankings)   # ranking indices with a \succ b

        if(len(vba) == 0):
            continue
        
        model = gp.Model("maximin")
        # model.Params.LogToConsole = 0
        
        P_x = dict()    
        x = dict()
        z = dict()
        temp = dict()
        
        for i, R_i in enumerate(vba):
            x[i] = model.addVar(vtype=gp.GRB.INTEGER, name=f"{rankings[vba[i]]}")
            model.addConstr(x[i] >= 0)
            model.addConstr(x[i] <= int(anon_votes[R_i][0]))
        
        for c in range(m):
            for d in range(m):
                
                if c == d:
                    continue
                
                P_x[(c, d)] = 0
                for i, R_i in enumerate(vba):
                    P_x[(c, d)] += x[i] * (prefab(c, d, rankings[R_i]) > 0)
                for i, R_i in enumerate(vab):
                    P_x[(c, d)] += int(anon_votes[R_i][0]) * (prefab(c, d, rankings[R_i]) > 0)
         
        cnt = 0
        
        z[b] = model.addVar(vtype = gp.GRB.INTEGER, name=f'score_{b}')
        for c in range(m):
            if b != c:
                temp[cnt] = model.addVar(vtype = gp.GRB.INTEGER, name=f'P_x[{b},{c}]')
                model.addConstr(temp[cnt] == P_x[(b, c)])
                cnt += 1
                model.addConstr(z[b] <= P_x[(b, c)])
        
        for c in range(m):
            if b == c:
                continue
            z[c] = model.addVar(vtype = gp.GRB.INTEGER, name=f'score_{c}')
        
            cnt_start = cnt
            for d in range(m):
                if d == c:
                    continue
                temp[cnt] = model.addVar(vtype = gp.GRB.INTEGER, name=f'P_x[{c},{d}]')
                model.addConstr(temp[cnt] == P_x[(c, d)])
                cnt += 1
                
            model.addConstr(z[c] == gp.min_([temp[i] for i in range(cnt_start, cnt)]))
            
            if b > c:
                model.addConstr(z[c] <= z[b] - 1)
            else:
                model.addConstr(z[c] <= z[b])
        
        model.setObjective(z[b], gp.GRB.MAXIMIZE)
        
        # model.write("myfile.lp")
        model.optimize()
           
        if debug:
            print(model.status)
        if model.status != gp.GRB.OPTIMAL:
            continue
        lp_flag = 1
        
        # need to fix this
        # k = sum([anon_votes[vv][0] for vv in vba]) + model.ObjVal

        if debug:
            print('=' * 10)
            print(f'original winner = {a}, possible winner = {b}')
            # print(f'k = {k}')
            pprint(anon_votes)
            # for v in model.getVars():
            #     print(v.varName, v.x)
            pprint(model.getVars())
            print('=' * 10)
            
            for b in range(m):
                for c in range(m):
                    if c == b:
                        continue
                    print(b, c, P_x[(b, c)].getValue())
                    
        # if k < k_min:
        #     k_min = k
    
    if lp_flag == 0:
    # if k_min == np.inf:
        return 0, None, None, lp_flag
    return 1, 1, None, lp_flag 

def create_LP_STV(anon_votes, b, vbw, vwb, order, rankings, debug = False):
    
    '''we check plurality winner for each stage
    keep track of eliminated alternatives
    
    e.g. elimination order [0,1,2,3]
    
    R1 s_1>=s_0, s_2>=s_0, s_3>=s_0 (0 eliminated)
    R2 s_2>=s_1, s_3>=s_1 (1 eliminated)
    R3 s_3>=s_2
    '''
    print_flag = False
    if(debug):
        if(order == [3, 2, 1, 0]):
            print_flag = True
            print(order)
    
    A = [] # A : need to look into vbw ^^
    h = [] # h: need to look into vwb
    
    m = len(rankings[0])
    
    eliminated = []
    
    for idx, j in enumerate(order):
        
        if(print_flag):
            print(eliminated)        
        if idx == m - 1:
            break
        for i in range(m):
            if i == j or i in eliminated:
                continue
            temp = []
            if(print_flag):
                    print('LHS:', j, i)
            for k in vbw:
                # update A
                temp.append(R_top_a(j, rankings[k], eliminated) - 
                            R_top_a(i, rankings[k], eliminated))
            A.append(temp)
            
            h0 = 0
            if(print_flag):
                print('RHS:', i, j)

            for k in vwb:
                # update h
                h0 += int(anon_votes[k][0]) * (R_top_a(i, rankings[k], eliminated) - 
                                               R_top_a(j, rankings[k], eliminated))
            
            if j >= i:
                if(print_flag):
                    print('<=')
                h.append(h0)
            else:
                if(print_flag):
                    print('<')
                h.append(h0 - 1)
        eliminated.append(j)
    
    for l, k in enumerate(vbw):
        A.append(e(len(vbw),l))
        h.append(int(anon_votes[k][0]))
        A.append(neg(e(len(vbw),l)))
        h.append(0)
    
    # print(A)
    # print(h)
    
    A = cvxopt.matrix(A, tc = 'd')
    h = cvxopt.matrix(h, tc = 'd')
    c = cvxopt.matrix([-1 for k in range(len(vbw))], tc = 'd')
    
    # print(f'{A.size=}, {h.size=}, {c.size=}')
    
    return A, h, c


    
def ILP_STV_lexicographic(votes, anon_votes, n, m, elimination_orders, debug = False):
    
    # initializtion
    w_, s = STV_winner(votes)
    w = lexicographic_tiebreaking(votes, w_)
    
    if(debug):
        print(f'\tSTV_winner:{w}')
    
    rankings = [av[1] for av in anon_votes]
    # print(votes)
    
    # building the LP
    
    # w is the winner
    # say b is the target winner
    
    k_min = np.inf
    x_min = None
    lp_flag = 0
    
    for order in (elimination_orders):
        b = order[-1] # the one that doesn't get eliminated wins
        if b == w:
            continue
        
        vbw = Rab(b, w, rankings)   # ranking indices with b \succ w
        vwb = Rab(w, b, rankings)   # ranking indices with w \succ b
        
        len_x = len(vbw) # no. of variables
        # print(f'{len(rankings)=}, {len_x=}')
        if(len(vbw) == 0):
            continue
        
        # form an LP
        #print(f'\t\t{pair}')
        A, h, c = create_LP_STV(anon_votes, b, vbw, vwb, order, rankings, False)
        
        sol = cvxopt.solvers.lp(c, A.T, h)
        # print(sum(c.T * sol['x']))
        if(sol['status'] != 'optimal'):
            continue
        
        #print('solution for LP', sol['x'])
        lp_flag = 1
        
        try:
            (status, x) = cvxopt.glpk.ilp(c, A.T, h, I=set(range(len(c))))
            if(status != 'optimal'):
                if(debug):
                    print('ILP not feasible', 'status')
                continue
            else:
                if(debug):
                    print('ILP feasible')
                    print('winner:',b)
                    print([rankings[vv] for vv in vbw])
                    print('original votes')
                    print([anon_votes[vv][0] for vv in vbw])
                    print('new votes?')
                    print(x)
                    
                # c.T*x is the number of voters who has voted, not who's abstained
                # so to get number of abstanation, we can do the following
                k = sum([anon_votes[vv][0] for vv in vbw]) + sum(c.T * x)
                
                print('k:', k)
                
                if k < k_min:
                    k_min = k
        except:
            print("Exception occured")
        
    if k_min == np.inf:
        return 0, k_min, lp_flag
    return 1, k_min, lp_flag

def ILP_STV_lexicographic_new(votes, anon_votes, n, m, debug = False):
    a_, s = STV_winner(votes)
    a = lexicographic_tiebreaking(votes, a_)
    
    print(f'winner={a}')
    rankings = [av[1] for av in anon_votes]
    
    k_min = np.inf
    lp_flag = 0

    for b in range(m):
        if b == a:
            continue

        vba = Rab(b, a, rankings)   # ranking indices with b \succ a
        vab = Rab(a, b, rankings)   # ranking indices with a \succ b

        if(len(vba) == 0):
            continue
        
        print(b)
        
        model = gp.Model("STV")
        
        # x[i] is number of people actually voting for R_i
        x = dict()
        # e_a,r = 1 if a is eliminated in round r
        e = dict()
        # score will hold the expression for scores
        score = dict()
        # score_v is an auxilliary variable that stores score
        score_v = dict()
        
        # bounds for x[i]
        for i, R_i in enumerate(vba):
            x[i] = model.addVar(vtype=gp.GRB.INTEGER, name=f"x_{i}")
            model.addConstr(x[i] >= 0)
            model.addConstr(x[i] <= int(anon_votes[R_i][0]))
        
        # create e and score_v variables
        for c in range(m):
            for r in range(1, m):
                e[(c, r)] = model.addVar(vtype = gp.GRB.BINARY, name = f'e_({c},{r})')
                score_v[(c, r)] = model.addVar(vtype = gp.GRB.INTEGER, name = f'score_({c},{r})')
        
        # temp will help with dynamic build up of the elimination status,
        #   the product term in the expression
        temp = dict()
        cnt = 0
        
        # first, we deal with vba rankings, where agents might abstain
        # so, we're using x[i]
        for i, R_i in enumerate(vba):
            # incrementing round 1 scores, no one eliminated yet
            if (rankings[R_i][0], 1) not in score:
                score[(rankings[R_i][0], 1)] = x[i]
            else:
                score[(rankings[R_i][0], 1)] += x[i]
            
            # incrementing round k scores for k = 2 to m-1
            # temp[cnt] will hold the products of \sum_e_{bj}
            for k in range(2, m):
                temp[cnt] = model.addVar(vtype = gp.GRB.INTEGER)
                model.addConstr(temp[cnt] == gp.quicksum(e[(rankings[R_i][0], j)] for j in range(1, k)))
                cnt += 1
                for kk in range(2, k):
                    temp[cnt] = model.addVar(vtype = gp.GRB.INTEGER)
                    nxt = gp.quicksum(e[(rankings[R_i][kk-1], j)] for j in range(1, k))
                    model.addConstr(temp[cnt] == temp[cnt - 1] * nxt)
                    cnt += 1
                if (rankings[R_i][k - 1], k) not in score:
                    score[(rankings[R_i][k - 1], k)] = temp[cnt - 1] * x[i]
                score[(rankings[R_i][k - 1], k)] += temp[cnt - 1] * x[i]
        
        # next, we do the exact same thing for vab rankings, agents bay not abstain
        # so no x[i], instead using n_i
        for i, R_i in enumerate(vab):
            n_i = int(anon_votes[R_i][0])
            if (rankings[R_i][0], 1) not in score:
                score[(rankings[R_i][0], 1)] = n_i
            else:
                score[(rankings[R_i][0], 1)] += n_i
            for k in range(2, m):
                temp[cnt] = model.addVar(vtype = gp.GRB.INTEGER)
                model.addConstr(temp[cnt] == gp.quicksum(e[(rankings[R_i][0], j)] for j in range(1, k)))
                cnt += 1
                for kk in range(2, k):
                    temp[cnt] = model.addVar(vtype = gp.GRB.INTEGER)
                    nxt = gp.quicksum(e[(rankings[R_i][kk-1], j)] for j in range(1, k))
                    model.addConstr(temp[cnt] == temp[cnt - 1] * nxt)
                    cnt += 1
                if (rankings[R_i][k - 1], k) not in score:
                    score[(rankings[R_i][k - 1], k)] = temp[cnt - 1] * n_i
                score[(rankings[R_i][k - 1], k)] += temp[cnt - 1] * n_i
        
        # creating the STV_scores for different rounds
        # multiplying (1-\sum (c,j)) for all alternatives
        # this will turn STV_scores for eliminated alternatives to zero
        STV_score = dict()
        for c in range(m):
            for r in range(1, m):
                if r == 1:
                    STV_score[(c, r)] = model.addVar(vtype = gp.GRB.INTEGER, name = f'STV_({c},{r})')
                    model.addConstr(STV_score[(c, r)] == score[(c, r)])
                else:
                    model.addConstr(score_v[(c,r)] == score[(c,r)])
                    STV_score[(c, r)] = model.addVar(vtype = gp.GRB.INTEGER, name = f'STV_({c},{r})')
                    survive =  1 - gp.quicksum(e[(rankings[R_i][0], j)] for j in range(1, k))
                    model.addConstr(STV_score[(c, r)] == score_v[(c, r)] * survive)
        
        # finally, adding the STV rule constraints
        # for b to be the winner, be needs to be not eliminated in each round
        # that is, STV_score for b has to be greater (or equal to) than the minumum
        #   score of that round
        tmp_lst = dict()
        cnt_lst = 0
        for r in range(1, m):
            # for each round, create list of scores first
            # two lists, one for items before b, one for after
            lst_before = []
            lst_after = []
            
            for c in range(m):
                if c == b:
                    continue
            
                # if first round, this is easier
                # since no one has been eliminated yet, just add the score of  
                #   each alt to the list
                if r == 1:
                    tmp_lst[cnt_lst] = model.addVar(vtype = gp.GRB.INTEGER)
                    model.addConstr(tmp_lst[cnt_lst] == STV_score[(c, r)])
                    if (c < b):
                        lst_before.append(tmp_lst[cnt_lst])   
                    else:
                        lst_after.append(tmp_lst[cnt_lst])
                    cnt_lst += 1
                # else, we have to do consider already eliminated alts, because
                #   we don't want to consider them
                # if elim == 0, STV_score would be zero, instead of that, just add
                #   a large number so the min_ operation isn't messed up
                # scores are upperb bounded by n, so that's a good idea
                else:
                    elim = gp.quicksum(e[(c, j)] for j in range(1, r))
                    tmp_lst[cnt_lst] = model.addVar(vtype = gp.GRB.INTEGER)
                    model.addConstr(tmp_lst[cnt_lst] == (1 - elim) * STV_score[(c, r)] + elim * n)
                    if (c < b):
                        lst_before.append(tmp_lst[cnt_lst])   
                    else:
                        lst_after.append(tmp_lst[cnt_lst])
                    cnt_lst += 1
                
            # add two constraints with mins
            # for lexicographic tiebreaking
            model.addConstr(STV_score[(b, r)] >= 1 + gp.min_(lst_before))
            model.addConstr(STV_score[(b, r)] >= gp.min_(lst_after))
        
        model.setObjective(-gp.quicksum(x[i] for i in range(len(vba))), gp.GRB.MINIMIZE)
        #model.write("STV_file.lp")
        

        model.optimize()
        if debug:
            print(model.status)
        if model.status != gp.GRB.OPTIMAL:
            continue
        lp_flag = 1
        
        k = sum([anon_votes[vv][0] for vv in vba]) + model.ObjVal

        if debug:
            print('=' * 10)
            print(f'original winner = {a}, possible winner = {b}')
            print(f'k = {k}')
            print(anon_votes)
            for v in model.getVars():
                print(v.varName, v.x)
            print('=' * 10)

        if k < k_min:
            k_min = k
    
    if k_min == np.inf:
        return 0, None, None, lp_flag
    return 1, k_min, None, lp_flag

def create_LP_Blacks(anon_votes, vbw, vwb, b, rankings):
    '''
    Returns
    -------
    Only create the newer conditions, return an np array.
    '''
    A1 = []
    h1 = []
    m = len(rankings[0])
    
    for a in range(m):
        if b == a:
            continue
        temp = []
        for k in vbw:
            temp.append(R_rank_a(a, rankings[k]) - R_rank_a(b, rankings[k]))
        A1.append(temp)
            
        h0 = 0
        for k in vwb:
            h0 += int(anon_votes[k][0]) * (R_rank_a(b, rankings[k]) - 
                                           R_rank_a(a, rankings[k]))
        if(b < a):
            h1.append(h0)
        else:
            h1.append(h0 - 1)
    
    # print(f'{len(vbw)}, {len(A1)=}, {len(h1)=}')
    return np.array(A1), np.array(h1)


def ILP_Blacks_lexicographic(votes, anon_votes, n, m, UMGS, edges, debug = False):
    
    # initializtion
    w_, s = Blacks_winner(votes)
    w = lexicographic_tiebreaking(votes, w_)
    
    rankings = [av[1] for av in anon_votes]
    # print(votes)
    
    # building the LP
    
    # w is the winner
    # say b is the target winner
    
    k_min = np.inf
    x_min = None
    lp_flag = 0
    
    # go through all possible umgs
    for g, umg in (enumerate(UMGS)):
        #print(f'UMG: {g}')
        b, scores = Copeland_winner_from_umg(umg, m, edges)
        if(np.max(scores) == m-1):
            # Same as Copeland
            # There is a Condorcet winner
        
            if b == w:
                continue
            
            # print('b', b, 'umg', umg)
            vbw = Rab(b, w, rankings)   # ranking indices with b \succ w
            vwb = Rab(w, b, rankings)   # ranking indices with w \succ b
            
            len_x = len(vbw) # no. of variables
            if(len(vbw) == 0):
                continue
            
            # check to see if b can be possible winner
            
            # form an LP
            A, h, c = create_LP_Copeland(anon_votes, vbw, vwb, edges, umg, rankings)
            
            sol = cvxopt.solvers.lp(c, A.T, h)
            # print(sum(c.T * sol['x']))
            if(sol['status'] != 'optimal'):
                continue
            lp_flag = 1
            
            (status, x) = cvxopt.glpk.ilp(c, A.T, h, I=set(range(len(c))))
            if(status != 'optimal'):
                if(debug):
                    print('ILP not feasible', 'status')
                continue
            else:
                if(debug):
                    print('ILP feasible')
                # c.T*x is the number of voters who has voted, not who's abstained
                # so to get number of abstanation, we can do the following
                k = sum([anon_votes[vv][0] for vv in vbw]) + sum(c.T * x)
                print('k:', k)
                
                if k < k_min:
                    k_min = k
                    # x_min = x
                    # print('rankings:', [rankings[vv] for vv in vbw])
                    # print('x:', x)
                    # new_votes = create_full_pref(create_new_pref(anon_votes, vbw, list(x)))
                    
                    new_w_, new_s = Copeland_winner(new_votes)
                    new_w = lexicographic_tiebreaking(votes, new_w_)
                    
                    print('found something', 'w', w, 'b', b, umg, 'Had Condorcet winner')
                    
        else:
            # No Condorcet winner
            # Need UMG-based constraints, but also other constraints
            # Break into a for loop for m alternatives
            
            for b in range(m):
                if b == w:
                    continue
                
                vbw = Rab(b, w, rankings)   # ranking indices with b \succ w
                vwb = Rab(w, b, rankings)   # ranking indices with w \succ b
                
                len_x = len(vbw) # no. of variables
                if(len(vbw) == 0):
                    continue    
                
                # create the Copeland LP
                A, h, c = create_LP_Copeland(anon_votes, vbw, vwb, edges, umg, rankings)
                # print(f'{A.size}, {h.size}')
                
                A = np.array(A)
                h = np.array(h)
                
                A1, h1 = create_LP_Blacks(anon_votes, vbw, vwb, b, rankings)
                
                # merge the two sets of constraints
                # print(f'{len_x=}', h.shape, h1.shape)
                # print(f'{len_x=}', A.shape, A1.shape)
                A = np.concatenate((A.T, A1))
                h = np.concatenate((h.flatten(), h1))
                A = cvxopt.matrix(A, tc = 'd')
                h = cvxopt.matrix(h, tc = 'd')
                
                # print(f'{A.size}, {h.size}')
                # print(A)
                # print(H)
                
                sol = cvxopt.solvers.lp(c, A, h)
                # print(sum(c.T * sol['x']))
                if(sol['status'] != 'optimal'):
                    continue
                lp_flag = 1
                
                (status, x) = cvxopt.glpk.ilp(c, A, h, I=set(range(len(c))))
                if(status != 'optimal'):
                    if(debug):
                        print('ILP not feasible', 'status')
                    continue
                else:
                    if(debug):
                        print('ILP feasible')
                    # c.T*x is the number of voters who has voted, not who's abstained
                    # so to get number of abstanation, we can do the following
                    k = sum([anon_votes[vv][0] for vv in vbw]) + sum(c.T * x)
                    print('k:', k)
                    
                    if k < k_min:
                        k_min = k
                        # x_min = x
                        # print('rankings:', [rankings[vv] for vv in vbw])
                        # print('x:', x)
                                
                        # new_votes = create_full_pref(create_new_pref(anon_votes, vbw, list(x)))
                    
                        # new_w_, new_s = Blacks_winner(new_votes)
                        # new_w = lexicographic_tiebreaking(votes, new_w_) 
                        
                        print('found something', 'w', w, 'b', b, 'No Condorcet winner')
    
    if k_min == np.inf:
        return 0, None, lp_flag
    return 1, k_min, lp_flag

def ILP_Blacks_lexicographic_new_2(votes, anon_votes, n, m, debug = False):
    a_, s = Blacks_winner(votes)
    a = lexicographic_tiebreaking(votes, a_)
    
    rankings = [av[1] for av in anon_votes]
    
    k_min = np.inf
    lp_flag = 0

    for b in range(m):
        if b == a:
            continue

        vba = Rab(b, a, rankings)   # ranking indices with b \succ a
        vab = Rab(a, b, rankings)   # ranking indices with a \succ b

        if(len(vba) == 0):
            continue

        model_blacks = gp.Model("blacks")
        x = dict()
        q = dict()
        r = dict()
        P_x = dict()
        score = dict()
        borda_score = dict()

        for i, R_i in enumerate(vba):
            x[i] = model_blacks.addVar(vtype=gp.GRB.INTEGER, name=f"x_{i}")
            model_blacks.addConstr(x[i] >= 0)
            model_blacks.addConstr(x[i] <= int(anon_votes[R_i][0]))

        for c in range(m):
            for d in range(c + 1, m):
                q[(c, d)] = model_blacks.addVar(vtype=gp.GRB.BINARY, name=f"q_{c}_{d}")
                q[(d, c)] = model_blacks.addVar(vtype=gp.GRB.BINARY, name=f"q_{d}_{c}")
                r[(c, d)] = model_blacks.addVar(vtype=gp.GRB.BINARY, name=f"r_{c}_{d}")
        
                model_blacks.addConstr(q[(c, d)] + r[(c, d)] + q[(d, c)] == 1)
        
                P_x[(c, d)] = 0
                P_x[(d, c)] = 0
                for i, R_i in enumerate(vba):
                    P_x[(c, d)] += x[i] * (prefab(c, d, rankings[R_i]) > 0)
                    P_x[(d, c)] += x[i] * (prefab(d, c, rankings[R_i]) > 0)
                for i, R_i in enumerate(vab):
                    P_x[(c, d)] += int(anon_votes[R_i][0]) * (prefab(c, d, rankings[R_i]) > 0)
                    P_x[(d, c)] += int(anon_votes[R_i][0]) * (prefab(d, c, rankings[R_i]) > 0)

                model_blacks.addConstr((q[(c, d)] == 1) >> (P_x[(c, d)] >= P_x[(d, c)] + 1))
                model_blacks.addConstr((q[(d, c)] == 1) >> (P_x[(d, c)] >= P_x[(c, d)] + 1))
                model_blacks.addConstr((r[(c, d)] == 1) >> (P_x[(c, d)] == P_x[(d, c)]))

        for c in range(m):
            score[c] = 0
            for d in range(m):
                if c == d:
                    continue
                # replace 0.5 by alpha
                score[c] += q[(c, d)] + 0.5 * r[(c, d) if c < d else (d, c)]

        for c in range(m):
            borda_score[c] = 0
            for i, R_i in enumerate(vba):
                borda_score[c] += x[i] * R_rank_a(c, rankings[R_i]) 
            for i, R_i in enumerate(vab):
                borda_score[c] += int(anon_votes[R_i][0]) * R_rank_a(c, rankings[R_i]) 
        
        # lexicographic tie-breaking
        is_Condorcet_winner = model_blacks.addVar(vtype=gp.GRB.BINARY, name=f"is_Condorcet_winner")
        is_Borda_winner = model_blacks.addVar(vtype=gp.GRB.BINARY, name=f"is_Borda_winner")
        b_is_winner = model_blacks.addVar(vtype=gp.GRB.BINARY, name=f"b_is_winner")

        model_blacks.addConstr((is_Condorcet_winner == 1) >> (score[b] == m - 1))

        for c in range(m):
            if b != c:
                model_blacks.addConstr(score[c] <= m - 1 - 1e-3) # score[c] < m - 1

            if b == c:
                continue
            elif b > c:
                model_blacks.addConstr((is_Borda_winner == 1) >> (borda_score[b] >= borda_score[c] + 1e-3)) # score[b] > score[c] but gurobi doesn't support strict inequality
            else:
                model_blacks.addConstr((is_Borda_winner == 1) >> (borda_score[b] >= borda_score[c]))

        model_blacks.addConstr(b_is_winner == gp.or_(is_Condorcet_winner, is_Borda_winner))
        model_blacks.addConstr(b_is_winner == 1)

        model_blacks.setObjective(-gp.quicksum(x[i] for i in range(len(vba))), gp.GRB.MINIMIZE)

        model_blacks.optimize()
        if debug:
            print('model_blacks', model_blacks.status)
        if model_blacks.status != gp.GRB.OPTIMAL:
            continue
        lp_flag = 1

        k = sum([anon_votes[vv][0] for vv in vba]) + model_blacks.ObjVal

        if debug:
            print('=' * 10)
            print(f'original winner = {a}, possible winner = {b}')
            black_w, _ = Blacks_winner(votes)
            print('Blacks winner', black_w)
            copeland_w, _ = Copeland_winner(votes)
            print('Copeland winner', copeland_w)
            borda_w, _ = Borda_winner(votes)
            print('Borda winner', borda_w)
            print(f'k = {k}')
            print(anon_votes)
            for v in model_blacks.getVars():
                print(v.varName, v.x)
            print('=' * 10)

        if k < k_min:
            k_min = k


    if k_min == np.inf:
        return 0, None, None, lp_flag
    return 1, k_min, None, lp_flag

# %%

def fixed_m_main():
    time_now = datetime.now().strftime('%y%m%d%H%M%S')
    
    n = 100
    m = 4
    distribution = 'IC'
    
    np.random.seed(0)
    
    # edges, UMGS = create_umgs(m)
    # print('UMGs created')
    
    # maximin_pairs = product_alt_pairs(m)
    # print('maximin pairs created', len(maximin_pairs))
    
    # perms = [list(p) for p in permutations(list(range(m)))]
    # print('permutations created')
    
    test = []
    test_ILP = []
    ILP_soln_all = []
    
    # for n in range(10, 11, 10):

    for n in [10, 100, 1000, 10000, 100000]:
        
        print(n)    
    
        brute_cnt = 0
        ILP_cnt = 0
        ILP_new_cnt = 0
        
        brute_times = []
        # ILP_succ_times = []
        # ILP_fail_times = []
        # ILP_soln = []
        ILP_new_succ_times = []
        ILP_new_fail_times = []
        ILP_new_soln = []
        
        Votes = []
        samples = 100
        
        # IC
        if(distribution == 'IC'):
            for s in tqdm(range(samples)):
                votes = gen_pref_profile(n, m)
                Votes.append(votes)
        # Mallows
        elif(distribution == 'Mallows'):
            W = np.array(range(m))
            phi = 0.5
            for s in tqdm(range(samples)):
                votes = gen_mallows_profile(n, W, phi)
                Votes.append(votes)
        
        for s in range(samples):
            # print(s)
            votes = Votes[s]
            m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
            # anon_votes = np.array(anon_votes)
            
            # brute_force_Copeland
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt += 1
            # tok = time()
            # print(tok - tik)
            # brute_times.append(tok-tik)
            
            # ILP_Copeland_old
            # tik = time()
            # flag, k, new_votes, lp_flag = ILP_Copeland_lexicographic(votes, anon_votes,
            #                                                           n, m, UMGS, edges, debug = False)
            # tok = time()
            # print(n, s, tok - tik)
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, new_votes, lp_flag])
            
            # ILP_new_Copeland
            tik = time()
            flag_ILP_new, k, new_votes, lp_flag = ILP_Copeland_lexicographic_new(votes, anon_votes,
                                                                      n, m, debug = False)
            tok = time()
            print(f'{(tok - tik)=}')
            if(flag_ILP_new):
                ILP_cnt += 1
                pp = copy.deepcopy(votes)
                anon_pp = copy.deepcopy(anon_votes)
                ILP_new_succ_times.append(tok-tik)
            else:
                ILP_new_fail_times.append(tok-tik)
            ILP_new_soln.append([flag_ILP_new, k, new_votes, lp_flag])                        
            
            # brute_force_maximin
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, maximin_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt += 1
            # tok = time()
            # print(tok - tik)
            # brute_times.append(tok-tik)
            
            
            # ILP_maximin
            # tik = time()
            # flag, k, lp_flag = ILP_maximin_lexicographic(votes, anon_votes, n, m, 
            #                                               maximin_pairs, debug = False)
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # tok = time()
            # print(f'{n=}, {s=},', tok - tik)
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, lp_flag])
            
            # ILP_STV
            # tik = time()
            # flag, k, lp_flag = ILP_STV_lexicographic(votes, anon_votes, n, m, 
            #                                               perms, debug = False)
            # tok = time()
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # print(n, s, tok - tik)
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, lp_flag])
            
            # brute_force_STV
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, STV_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt += 1
            # tok = time()
            # print(tok - tik)
            # brute_times.append(tok-tik)
            
            # ILP_Blacks
            # tik = time()
            # flag, k, lp_flag = ILP_Blacks_lexicographic(votes, anon_votes,
            #                                             n, m, UMGS, edges, debug = False)
            # tok = time()
            # print(n, s, tok - tik)
            # if(flag):
            #     ILP_cnt += 1
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, lp_flag])
            
            
        # print(brute_cnt, np.mean(brute_times))
        # test.append([n, brute_cnt, brute_times])
        # print(ILP_cnt, np.mean(ILP_times))
        # test_ILP.append([n, ILP_cnt, ILP_times])
        # ILP_soln_all.append(ILP_soln)
        
        print('===============================')
        print(f'{n=}')
        # print(brute_cnt, np.mean(brute_times))
        # test.append([n, brute_cnt, brute_times])

        print('ILP_new', ILP_cnt, np.mean(ILP_new_succ_times), np.mean(ILP_new_fail_times))
        test_ILP.append([n, ILP_cnt, ILP_new_succ_times, ILP_new_fail_times])
        ILP_soln_all.append(ILP_new_soln)

        print('===============================')
    
    
    for output in test_ILP:
        print('===============================')
        print(f'n={output[0]}')
        
        print('ILP_new', output[1], np.mean(output[2] + output[3]))
        print('===============================')
        
    
    if(distribution == 'Mallows'):
        distribution_print = f'Mallows_{phi}'
    else:
        distribution_print = distribution
    
    return test_ILP
    # with open(f'{time_now}_STV_ILP_GP_{m=}_{distribution_print=}_{samples=}.npy', 'wb') as f:
    #     np.save(f, np.array(test_ILP, dtype = object))
    #     np.save(f, np.array(ILP_soln_all, dtype = object))

    # with open(f'{time_now}_STV_brute_GP_{m=}_{distribution=}_{samples=}.npy', 'wb') as f:
    #     np.save(f, np.array(test, dtype = object))
   
    
def fixed_n_main():
    time_now = datetime.now().strftime('%y%m%d%H%M%S')
    
    n = 100
    distribution = 'IC'
    
    np.random.seed(0)
    
    test = []
    test_ILP = []
    ILP_soln_all = []
    test_ILP_new = []
    ILP_new_soln_all = []

    # count brute_force_times
    for m in range(12, 21, 4):
    
        print(f'{m=}')    

        # edges, UMGS = create_umgs(m)
        # print('UMGs created')

        brute_cnt = np.zeros(4)
        ILP_cnt = 0
        ILP_new_cnt = 0
        
        brute_times = []
        # ILP_times = []
        # ILP_soln = []
        # ILP_new_times = []
        # ILP_new_soln = []
        ILP_new_succ_times = []
        ILP_new_fail_times = []
        ILP_new_soln = []
        


        Votes = []
        samples = 50
        
        # IC
        if(distribution == 'IC'):
            for s in range(samples):
                votes = gen_pref_profile(n, m)
                Votes.append(votes)
        # Mallows
        elif(distribution == 'Mallows'):
            W = np.array(range(m))
            phi = 0.8
            for s in range(samples):
                votes = gen_mallows_profile(n, W, phi)
                Votes.append(votes)
        
        for s in (range(samples)):
            # print(s)
            votes = Votes[s]
            m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
            # anon_votes = np.array(anon_votes)
            
            # brute_force_Copeland
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt[0] += 1
            # tok = time()
            # # print(tok - tik)
            # Copeland_DFS_time = (tok-tik)
            
            # ILP_maximin new
            # tik = time()
            # flag, k, new_votes, lp_flag = ILP_maximin_lexicographic_new(votes, anon_votes,
            #                                                           n, m, debug = False)
            # tok = time()
            # # print(n, s, tok - tik)
            # if(flag):
            #     ILP_new_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # ILP_new_times.append(tok-tik)
            # ILP_new_soln.append([flag, k, new_votes, lp_flag])

            # ILP_Copeland
            # tik = time()
            # flag, k, new_votes, lp_flag = ILP_Copeland_lexicographic(votes, anon_votes,
            #                                                           n, m, UMGS, edges, debug = False)
            # tok = time()
            # # print(n, s, tok - tik)
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, new_votes, lp_flag])
            
            # ILP Copeland new
            
            tik = time()
            flag_ILP_new, k, new_votes, lp_flag = ILP_Copeland_lexicographic_new(votes, anon_votes,
                                                                      n, m, debug = False)
            tok = time()
            # print(f'{(tok - tik)=}')
            if(flag_ILP_new):
                ILP_cnt += 1
                pp = copy.deepcopy(votes)
                anon_pp = copy.deepcopy(anon_votes)
                ILP_new_succ_times.append(tok-tik)
            else:
                ILP_new_fail_times.append(tok-tik)
            ILP_new_soln.append([flag_ILP_new, k, new_votes, lp_flag])                        

            # # brute_force_maximin
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, maximin_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt[1] += 1
            # tok = time()
            # # print(tok - tik)
            # maximin_DFS_time = (tok-tik)
            
            
            # # brute_force_STV
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, STV_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt[2] += 1
            # tok = time()
            # print(tok - tik)
            # STV_DFS_time = (tok-tik)
            
            # # brute_force_Black
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Blacks_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt[3] += 1
            # tok = time()
            # print(tok - tik)
            # Blacks_DFS_time = (tok-tik)
            
            
            # brute_times.append([Copeland_DFS_time, maximin_DFS_time, 
            #                     STV_DFS_time, Blacks_DFS_time])
            # brute_times.append(maximin_DFS_time)
            
        print('===============================')
        print(f'{m=}')
        # print(brute_cnt, np.mean(brute_times))
        # test.append([n, brute_cnt, brute_times])

        print('ILP_new', ILP_cnt, np.mean(ILP_new_succ_times), np.mean(ILP_new_fail_times))
        test_ILP.append([m, ILP_cnt, ILP_new_succ_times, ILP_new_fail_times])
        ILP_soln_all.append(ILP_new_soln)

        print('===============================')
        # print('===============================')
        # print(f'{m=}')
        # print('brute', brute_cnt[0], np.mean(brute_times))
        # test.append([m, brute_cnt, brute_times])

        # # print('ILP', ILP_cnt, np.mean(ILP_times))
        # # test_ILP.append([m, ILP_cnt, ILP_times])
        # # ILP_soln_all.append(ILP_soln)

        # print('ILP_new', ILP_new_cnt, np.mean(ILP_new_times))
        # test_ILP_new.append([m, ILP_new_cnt, ILP_new_times])
        # ILP_new_soln_all.append(ILP_new_soln)
        # print('===============================')
    
        if(distribution == 'Mallows'):
            distribution_print = f'Mallows_{phi}'
        else:
            distribution_print = distribution
        
        # with open(f'DFS_CP_{m=}_{distribution_print=}_{samples=}_2.npy', 'wb') as f:
        #     np.save(f, np.array(test, dtype = object))
    
        # with open(f'ILP_CP_{m=}_{distribution_print=}_{samples=}.npy', 'wb') as f:
        #     np.save(f, np.array(test_ILP, dtype = object))
        #     np.save(f, np.array(ILP_soln_all, dtype = object))

        # with open(f'ILP_new_CP_{m=}_{distribution_print=}_{samples=}_2.npy', 'wb') as f:
        #     np.save(f, np.array(test_ILP_new, dtype = object))
            # np.save(f, np.array(ILP_new_soln_all, dtype = object))
    for output in test_ILP:
        print('===============================')
        print(f'm={output[0]}')
        
        print('ILP_new', output[1], np.mean(output[2] + output[3]))
        print('===============================')

def test_stv_correctness():
    n = 10
    m = 3
    votes = gen_pref_profile(n, m)
    m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
    pprint(anon_votes)
    ILP_STV_lexicographic_new(votes, anon_votes, n, m)
        
def test_gp_correctness():
    n = 50
    distribution = 'IC'
    
    np.random.seed(0)
    
    test = []
    test_ILP = []
    ILP_soln_all = []
    test_ILP_new = []
    ILP_new_soln_all = []

    # count brute_force_times
    m = 4
    print(f'{m=}')    

    # edges, UMGS = create_umgs(m)
    # print('UMGs created')
    
    maximin_pairs = product_alt_pairs(m)
    print('maximin pairs created', len(maximin_pairs))

    brute_cnt = np.zeros(4)
    ILP_cnt = 0
    ILP_new_cnt = 0
    
    ILP_soln = []
    ILP_new_soln = []
    
    ILP_times = []
    ILP_new_times = []

    Votes = []
    samples = 100
    
    # IC
    if(distribution == 'IC'):
        for s in range(samples):
            votes = gen_pref_profile(n, m)
            Votes.append(votes)
    # Mallows
    elif(distribution == 'Mallows'):
        W = np.array(range(m))
        phi = 0.9
        for s in range(samples):
            votes = gen_mallows_profile(n, W, phi)
            Votes.append(votes)
    
    # with gp.Env(empty=True) as env:
    #     # env.setParam("WLSAccessID", str)
    #     # env.setParam("WLSSECRET", str)
    #     # env.setParam("LICENSEID", int)
    #     env.setParam("OutputFlag", 0)
    #     env.start()
    
    for s in range(samples):
        print(10*"-",f"sample {s}",10*"-")
        votes = Votes[s]
        m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
        print(maximin_winner(votes))
        
        # brute_force_Copeland
        # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, 
        #                 lexicographic_tiebreaking)):
        #     brute_cnt[0] += 1
        
        # ILP_Copeland new
        # flag, k, new_votes, lp_flag = ILP_Copeland_lexicographic_new(votes, anon_votes,
        #                                                             n, m, UMGS, edges, debug = False)
        
        # ILP_maximin_new
        tik = time()
        flag, k, new_votes, lp_flag = ILP_maximin_lexicographic_new(votes, anon_votes,
                                                                    n, m, debug = True)
        tok = time()
        ILP_new_times.append(tok - tik)
        
        if(flag):
            ILP_new_cnt += 1
            pp = copy.deepcopy(votes)
            anon_pp = copy.deepcopy(anon_votes)
        ILP_new_soln.append([flag, k, new_votes, lp_flag])

        # ILP_Copeland
        # flag2, k2, new_votes2, lp_flag2 = ILP_Copeland_lexicographic(votes, anon_votes,
        #                                                             n, m, UMGS, edges, debug = False)
        
        # ILP_Maximin
        tik = time()
        flag2, k2, lp_flag2 = ILP_maximin_lexicographic(votes, anon_votes, n, m, 
                                                          maximin_pairs, debug = False)
        tok = time()
        ILP_times.append(tok - tik)
        
        if(flag2):
            ILP_cnt += 1
            pp = copy.deepcopy(votes)
            anon_pp = copy.deepcopy(anon_votes)
        # ILP_soln.append([flag2, k2, new_votes2, lp_flag2])
        ILP_soln.append([flag2, k2, lp_flag2])

        
        if flag != flag2:
            print(s, '='*20)
            print('flag not equal')
            print(flag, flag2)
            print(s, '='*20)
            break
        

        # if flag == flag2 and flag == True and k != k2:
        #     print('='*20)
        #     print('No.', s)
        #     print('k not equal')
        #     print('ILP_new', k, ', ILP', k2)
        #     w_, s = Copeland_winner(votes)
        #     w = lexicographic_tiebreaking(votes, w_)
        #     print('original winner', w, s)
        #     print('vote', anon_votes)
        #     # w_, s = Copeland_winner(new_votes2)
        #     # w = lexicographic_tiebreaking(new_votes2, w_)
        #     # print('new winner', w, s)
        #     # m, n, n_votes, n_unique, anon_votes2 = anonymize_pref_profile(new_votes2)
        #     # print('new vote', anon_votes2)
        #     print('='*20)
        #     break
    
    print('flag count and time for ILP_new')
    print(sum(row[0] for row in ILP_new_soln))
    print(np.mean(ILP_new_times))
    
    print('flag count for ILP_old')
    print(sum(row[0] for row in ILP_soln))
    print(np.mean(ILP_times))

if __name__ == '__main__':
    
    # test_ILP = fixed_m_main()
    fixed_n_main()
    