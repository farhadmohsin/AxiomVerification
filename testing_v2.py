import numpy as np
from fixed_m_anonymous_r_polynomial import algo, Copeland_winner_anon
from bruteforce import *
from ilp_gp import *
from time import time

if __name__ == '__main__':
        
    n = 20
    m = 4
    
    voting_rule = Copeland_winner
    tiebreaking = lexicographic_tiebreaking
    
    np.random.seed(0)
    
    brute_cnt = 0
    anon_brute_cnt = 0
    ILP_cnt = 0
    
    cnt = 0 
    
    brute_times = []
    anon_brute_times = []
    ILP_times = []
    
    samples = 100
    for s in range(samples):
        print(cnt)
        cnt += 1
        votes = gen_pref_profile(n, m)
        m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
        # anon_votes = np.array(anon_votes)
        
        # tik = time()
        # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, voting_rule, tiebreaking)):
        #     brute_cnt += 1
        # tok = time()
        # brute_times.append(tok-tik)
        
        tik = time()
        if(ILP_search(votes, anon_votes, n, m, voting_rule, tiebreaking)):
            ILP_cnt += 1
        tok = time()
        ILP_times.append(tok-tik)    
        
        tik = time()
        if(algo(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, tiebreaking, verbose = True)):
            anon_brute_cnt += 1
        tok = time()
        anon_brute_times.append(tok-tik)    
        
        # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, voting_rule, tiebreaking)):
        #     if(not(ILP_search(votes, anon_votes, n, m, voting_rule, tiebreaking))):
                # break
        
    print(brute_cnt, ILP_cnt, anon_brute_cnt)
    print(np.mean(brute_times), np.mean(ILP_times), np.mean(anon_brute_times))
