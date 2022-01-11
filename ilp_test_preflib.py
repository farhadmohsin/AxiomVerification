import numpy as np
from ilp_gp_new import *
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

# %%
if __name__ == '__main__':
    
    time_now = datetime.now().strftime('%y%m%d%H%M%S')
    
    
    # maximin_pairs = product_alt_pairs(m)
    # print('maximin pairs created', len(maximin_pairs))
    
    test = []
    test_ILP = []
    ILP_soln_all = []
    
    brute_cnt = 0
    ILP_cnt = 0
    
    brute_times = []
    ILP_times = []
    ILP_soln = []
    
    file_set1 = set([])
    # file_set2 = set([])
    
    for root, dirs, files in os.walk("./dataset/"):
        for file in files:
            
            m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc("./dataset/" + file)
            
            # for brute_force
            if(n >= 40):
                continue
            if(m >= 200):
                continue
            file_set1.add(file)
            
            # for ILP
            # if (m>=6):
            #     continue
        
            # file_set2.add(file)
            # print(file, m, n) 
            
            # brute_force
            tik = time()
            if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, Blacks_winner, 
                            lexicographic_tiebreaking)):
                brute_cnt += 1
            tok = time()
            print(tok - tik)
            brute_times.append(tok-tik)
            
            # ILP_Copeland/Black
            # edges, UMGS = create_umgs(m)
            # print('UMGs created')
            # tik = time()
            # flag, k, lp_flag = ILP_Blacks_lexicographic(votes, anon_votes,
            #                                             n, m, UMGS, edges, debug = False)
            # tok = time()
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            # print(tok - tik)
            # ILP_times.append(tok-tik)
            # ILP_soln.append([flag, k, lp_flag])
            
            # ILP_STV
            # perms = [list(p) for p in permutations(list(range(m)))]
            # print('permutations created')
            # tik = time()
            # flag, k, lp_flag = ILP_STV_lexicographic(votes, anon_votes, n, m, 
            #                                               perms, debug = False)
            # tok = time()
            # if(flag):
            #     ILP_cnt += 1
            #     pp = copy.deepcopy(votes)
            #     anon_pp = copy.deepcopy(anon_votes)
            #print(tok - tik)
            ILP_times.append(tok-tik)
            ILP_soln.append([flag, k, lp_flag])
            
        # TODO: ILP_maximin
        # tik = time()
        # flag, k, lp_flag = ILP_maximin_lexicographic(votes, anon_votes, n, m, 
        #                                               maximin_pairs, debug = False)
        # if(flag):
        #     ILP_cnt += 1
        #     pp = copy.deepcopy(votes)
        #     anon_pp = copy.deepcopy(anon_votes)
        # tok = time()
        # print(tok - tik)
        # ILP_times.append(tok-tik)
        # ILP_soln.append([flag, k, lp_flag])
        
        
    print(brute_cnt, np.mean(brute_times))
    # test.append([n, brute_cnt, brute_times])
    # print(ILP_cnt, np.mean(ILP_times))
    # test_ILP.append([ILP_cnt, ILP_times])
    # ILP_soln_all.append(ILP_soln)
        
    # with open(f'{time_now}_preflib_Blacks_ILP_GP.npy', 'wb') as f:
    #     np.save(f, np.array(test_ILP, dtype = object))
    #     np.save(f, np.array(ILP_soln_all, dtype = object))

