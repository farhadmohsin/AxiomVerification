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
                
                # TODO: we probably should rerun Copeland stuff after commenting this out
                new_votes = create_full_pref(create_new_pref(anon_votes, vbw, list(x)))
                votes_min = new_votes
                
                new_w_, new_s = Copeland_winner(new_votes)
                new_w = lexicographic_tiebreaking(votes, new_w_)
                print('found something', 'w', w, 'b', b, 'new_w', new_w, umg)
                # print(len(votes), len(new_votes))
                # print((votes))
                # print((new_votes))
                print('original scores:', s)
                print('new_scores:', new_s)
    
    if k_min == np.inf:
        return 0, None, None, lp_flag
    return 1, k_min, votes_min, lp_flag

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
    
    #TODO: try with cvxpy and SCIP
    
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
            x1 = cp.Variable(len_x)
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
    
    for order in tqdm(elimination_orders):
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
        
        if(debug and False):
            if(order == [3, 2, 1, 0]):
                print(f'{len(rankings)=}, {len_x=}')
                print([rankings[rank_idx] for rank_idx in vbw])
                print(f'{b=}')
                print(f'{order=}')
                print(f'{A.size=}, {h.size=}')
                # print(A)
                # print(h)
                # print(c)
        
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
                    
                    # new_w_, new_s = Copeland_winner(new_votes)
                    # new_w = lexicographic_tiebreaking(votes, new_w_)
                    
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


# %%

def fixed_m_main():
    time_now = datetime.now().strftime('%y%m%d%H%M%S')
    
    n = 100
    m = 4
    distribution = 'IC'
    
    np.random.seed(0)
    
    edges, UMGS = create_umgs(m)
    print('UMGs created')
    
    # maximin_pairs = product_alt_pairs(m)
    # print('maximin pairs created', len(maximin_pairs))
    
    # perms = [list(p) for p in permutations(list(range(m)))]
    # print('permutations created')
    
    test = []
    test_ILP = []
    ILP_soln_all = []
    
    for n in range(10, 101, 10):
    
        print(n)    
    
        brute_cnt = 0
        ILP_cnt = 0
        
        brute_times = []
        ILP_times = []
        ILP_soln = []
        
        Votes = []
        samples = 1000
        
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
        
        for s in range(samples):
            # print(s)
            votes = Votes[s]
            m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
            # anon_votes = np.array(anon_votes)
            
            # TODO: Improve the readibility here
            # TODO: Add in command line functionalities
            
            # brute_force_Copeland
            # tik = time()
            # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, 
            #                 lexicographic_tiebreaking)):
            #     brute_cnt += 1
            # tok = time()
            # print(tok - tik)
            # brute_times.append(tok-tik)
            
            # ILP_Copeland
            # tik = time()
            # flag, k, new_votes, lp_flag = ILP_Copeland_lexicographic(votes, anon_votes,
            #                                                           n, m, UMGS, edges, debug = False)
            # tok = time()
            # # print(tok - tik)
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, new_votes, lp_flag])
                        
            
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
            # print(f'{phi=}, {n=}, {s=},', tok - tik)
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, lp_flag])
            
            # ILP_STV
            tik = time()
            # flag, k, lp_flag = ILP_STV_lexicographic(votes, anon_votes, n, m, 
            #                                               perms, debug = False)
            # tok = time()
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # #print(tok - tik)
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
            tik = time()
            flag, k, lp_flag = ILP_Blacks_lexicographic(votes, anon_votes,
                                                        n, m, UMGS, edges, debug = False)
            tok = time()
            print(n, s, tok - tik)
            if(flag):
                ILP_cnt += 1
            ILP_times.append(tok-tik)
            ILP_soln.append([flag, k, lp_flag])
            
            
        # print(brute_cnt, np.mean(brute_times))
        # test.append([n, brute_cnt, brute_times])
        print(ILP_cnt, np.mean(ILP_times))
        test_ILP.append([n, ILP_cnt, ILP_times])
        ILP_soln_all.append(ILP_soln)
    
    if(distribution == 'Mallows'):
        distribution_print = f'Mallows_{phi}'
    else:
        distribution_print = distribution
    
    with open(f'{time_now}_Blacks_ILP_GP_{m=}_{distribution_print=}_{samples=}.npy', 'wb') as f:
        np.save(f, np.array(test_ILP, dtype = object))
        np.save(f, np.array(ILP_soln_all, dtype = object))

    # with open(f'{time_now}_STV_brute_GP_{m=}_{distribution=}_{samples=}.npy', 'wb') as f:
    #     np.save(f, np.array(test, dtype = object))
   
    
def fixed_n_main():
    time_now = datetime.now().strftime('%y%m%d%H%M%S')
    
    n = 10
    distribution = 'IC'
    
    np.random.seed(0)
    
    # count brute_force_times
    for m in range(4, 21, 2):
    
        print(f'{m=}')    
    
        brute_cnt = np.zeros(4)
        
        brute_times = []
        
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
        
        for s in range(samples):
            # print(s)
            votes = Votes[s]
            m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
            # anon_votes = np.array(anon_votes)
            
            # TODO: turn this into a for loop for readability
            
            # brute_force_Copeland
            tik = time()
            if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, 
                            lexicographic_tiebreaking)):
                brute_cnt[0] += 1
            tok = time()
            print(tok - tik)
            Copeland_DFS_time = (tok-tik)
            
            # brute_force_maximin
            tik = time()
            if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, maximin_winner, 
                            lexicographic_tiebreaking)):
                brute_cnt[1] += 1
            tok = time()
            print(tok - tik)
            maximin_DFS_time = (tok-tik)
            
            # brute_force_STV
            tik = time()
            if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, STV_winner, 
                            lexicographic_tiebreaking)):
                brute_cnt[2] += 1
            tok = time()
            print(tok - tik)
            STV_DFS_time = (tok-tik)
            
            # brute_force_Black
            tik = time()
            if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Blacks_winner, 
                            lexicographic_tiebreaking)):
                brute_cnt[3] += 1
            tok = time()
            print(tok - tik)
            Blacks_DFS_time = (tok-tik)
            
            
            brute_times.append([Copeland_DFS_time, maximin_DFS_time, 
                                STV_DFS_time, Blacks_DFS_time])
            
            
        # print(brute_cnt, np.mean(brute_times))
        # test.append([n, brute_cnt, brute_times])
    
        if(distribution == 'Mallows'):
            distribution_print = f'Mallows_{phi}'
        else:
            distribution_print = distribution
        
        with open(f'{time_now}_DFS_all_{m=}_{distribution_print=}_{samples=}.npy', 'wb') as f:
            np.save(f, np.array(brute_times, dtype = object))
            np.save(f, np.array(brute_cnt, dtype = object))
    
        # with open(f'{time_now}_STV_brute_GP_{m=}_{distribution=}_{samples=}.npy', 'wb') as f:
        #     np.save(f, np.array(test, dtype = object))
        


if __name__ == '__main__':
    
    fixed_m_main()
    # fixed_n_main()
