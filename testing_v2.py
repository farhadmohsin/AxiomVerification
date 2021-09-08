import numpy as np
from fixed_m_anonymous_r_polynomial import algo, Copeland_winner_anon
from bruteforce import *
from ilp_gp import *
from time import time
from datetime import datetime
from tqdm import tqdm

if __name__ == '__main__':
        
    n = 20
    m = 4
    
    voting_rules = [Copeland_winner]
    tiebreaking_methods = [lexicographic_tiebreaking, voter1_tiebreaking,
                            singleton_lex_tiebreaking, singleton_v1_tiebreaking]
    # tiebreaking_methods = [singleton_lex_tiebreaking]
    
    np.random.seed(0)
    
    
    samples = 100
    
    for n in range(20, 21, 5):
        
        print(n, m, samples)
        
        tik = time()
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        # save  the preference profile
        sample_votes = [gen_pref_profile(n, m) for s in range(samples)]
        sample_votes = np.array(sample_votes)
        
        # save the output vals here
        with open(f'{n}-{m}-{time_str}-pref_profile.npy', 'wb') as f:
            np.save(f, sample_votes)
        
        for voting_rule in voting_rules:
            for tiebreaking in tiebreaking_methods:
                
                
                brute_cnt = 0
                anon_brute_cnt = 0
                ILP_cnt = 0
                
                brute_times = []
                anon_brute_times = []
                ILP_times = []
                
                tik1 = time()
                for i, votes in enumerate(sample_votes):
                    # print(s)
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
                    ILP_times.append(['ILP',ILP_cnt, tok-tik, voting_rule.__name__, tiebreaking.__name__])    
                    
                    tik = time()
                    if(algo(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner, tiebreaking, verbose = True)):
                        anon_brute_cnt += 1
                    tok = time()
                    anon_brute_times.append(['anon_brute',anon_brute_cnt, tok-tik, voting_rule.__name__, tiebreaking.__name__])    
                    
                    # if(algo(m, n, n_votes, n_unique, votes, anon_votes, voting_rule, tiebreaking)):
                    #     if(not(ILP_search(votes, anon_votes, n, m, voting_rule, tiebreaking))):
                    #         break
                
                with open(f'{n}-{m}-{time_str}-{voting_rule.__name__}-{tiebreaking.__name__}-participation_sat.npy', 'wb') as f:
                    np.save(f, ILP_times)
                    np.save(f, anon_brute_times)
            
                print(voting_rule.__name__, tiebreaking.__name__, ILP_cnt, anon_brute_cnt)
                # print(np.mean(ILP_times), np.mean(anon_brute_times))
                
                tok1 = time()
                print("The whole thing took: ", tok1 - tik1)
