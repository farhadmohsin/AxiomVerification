import numpy as np
from voting_utils import *
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
from collections import deque
from copy import deepcopy
#from ilp_gp_new import prefab

#brute_force(m, n, n_votes, n_unique, votes, anon_votes, maximin_winner, 
            #                 lexicographic_tiebreaking)

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

def anon_brute_force(m, n, n_votes, n_unique, votes, anon_votes, r, tiebreaking):
    """
    Parameters
    ----------
    m : number of alternatives.
    n : number of agents.
    n_unique : num.
    votes : preference profile, list of votes, length [n].
    anon_votes : anonymized version of votes, count for each ranking, length <= [m!].
    r : voting rule.
    tiebreaking : tie-breaking rule.

    Returns
    -------
    True if GNSP occurs for this pref profile
    stdout: original winner, winner after gnsp occurence
    """
    w_, s_ = r(votes)
    w = tiebreaking(votes, w_)
    
    flag = True
    
    tik = time()
    
    for b in range(m):
        if b == w:
            continue
        
        q = deque()
        # perm_idx indicates index of the permutation, in range [0, n_unique]
        perm_idx = 0
        # queue (q) is initialized with initial profile and first permutation
        q.append((anon_votes, perm_idx))
        
        # debugging variable
        node_cnt = 1
        
        # breadth-first-search
        # goal state: a case of gnsp
        while(q):
            node_cnt += 1
            anon_top, perm_idx = q.popleft()
            
            # checking if new profile indicates no-show paradox with b as new winner
            votes_new = create_full_pref(anon_top)
            w_, s_ = r(votes_new)
            w_new = tiebreaking(votes_new, w_)
            
            if w_new == b:
                flag = False
                break
            
            # iterate through rankings to add to BFS queue
            for idx in range(perm_idx, n_unique): #n_unique = len(anon_votes)
                
                perm = anon_top[idx]
                if prefab(w, b, perm[1]) > 0:
                    continue
                
                anon_next = deepcopy(anon_top)
                if perm[0] == 0:
                    continue
                else:
                    anon_next[idx] = [perm[0] - 1, perm[1]]
                    q.append((anon_next, idx))
        
        if not flag:
            print(f'{w=}, {b=}')
            break
    
    return not flag

if __name__ == '__main__':

    times_all = []
    times_mean = []

    results_all = []

    samples = 10

    r = Blacks_winner
    tb = lexicographic_tiebreaking
        

    for n in [60]:

        #n = 20
        m = 4
        distribution = 'IC'
        
        np.random.seed(0)
        
        Votes = []
        # IC
        if(distribution == 'IC'):
            for s in range(samples):
                votes = gen_pref_profile(n, m)
                Votes.append(votes)
        # Mallows
        elif(distribution == 'Mallows'):
            W = np.array(range(m))
            phi = 0.5
            for s in range(samples):
                votes = gen_mallows_profile(n, W, phi)
                Votes.append(votes)
            
        cnt = 0
        times = []
        
        results = []
        
        for s in range(samples):
            votes = Votes[s]
            m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
            
            brute_flag = False
            brute_time = 0
            # tik = time()
            # brute_flag = brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner,
            #               lexicographic_tiebreaking)
            # tok = time()
            # brute_time = tok - tik
            
            
            tik = time()
            flag = anon_brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, 
                                    lexicographic_tiebreaking)
            tok = time()
            
            if flag:        
                cnt += 1
                
            results.append([flag, brute_flag])
            print(f'{n=},{s=}', tok - tik)
            times.append([tok - tik, brute_time])
                
            #for now
            # if not flag:
            #     break
        
        times_all.append([n , m, times])
        times_mean.append([n, m] + list(np.mean(times, axis = 0)))
        results_all.append([n, m] + list(np.mean(results, axis = 0)))
        
        print(f'{cnt=}, {samples=}')