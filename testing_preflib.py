import numpy as np
from preflib_utils import read_preflib_soc
from voting_utils import *
from bruteforce import *
from time import time
from datetime import datetime
from tqdm import tqdm
from testing import Condorcet_sat, Condorcet_saitsfaction, singleton_exists, \
    neutrality_anonymity_satisfaction
import pandas as pd

#%%

if __name__ == "__main__":
    
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    df = pd.DataFrame()
    
    voting_rules = [plurality_winner, Borda_winner, veto_winner, 
                    Copeland_winner, maximin_winner]
    # tiebreaking_methods = [lexicographic_tiebreaking, voter1_tiebreaking,
    #                         singleton_lex_tiebreaking, singleton_v1_tiebreaking]
    # voting_rules = [Copeland_winner]
    tiebreaking_methods = [lexicographic_tiebreaking, singleton_lex_tiebreaking]
    
    tie_pairs = list(zip(['lexicographic_tiebreaking', 'voter1_tiebreaking',
                           'singleton_lex_tiebreaking', 'singleton_v1_tiebreaking'],
                         ['Neutrality', 'Anonymity', 'Neutrality', 'Anonymity']))
    
    #  sampling preference profiles
    
    debug_flag = False
    for root, dirs, files in os.walk("./dataset/"):
        for f_no,file in enumerate(files):
            if "ED-00004" in file:
                continue
            
            if(debug_flag):
                break
            
            tik = time()
            
            vals = []
            
            m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc("./dataset/" + file)
            print(f_no, file, m, n)
            
            for voting_rule in voting_rules:
                
                # print(voting_rule.__name__)
                for tiebreaking in tiebreaking_methods:
                    
                    # print(tiebreaking.__name__)
                    # Condorcet satisfaction for various tiebreaking
                    exist, cond_satisfaction = Condorcet_sat(votes, voting_rule, tiebreaking)
                    if(exist):
                        sat_val = cond_satisfaction
                    else:
                        sat_val = 1
                    vals.append([file, m, n, 'CC', voting_rule.__name__, tiebreaking.__name__,
                                sat_val])
                    
                # if(voting_rule.__name__ == "Borda_winner"):
                #     if(not(sat_val)):
                #         debug_flag = True
                #         break
                
                # print("CC done")
                
                # Neutrality, anonimity for relevant tiebreaking
                # w, s = voting_rule(votes)
                
                # lex_neutral = 1
                # lex_mpsr_neutral = 1
                
                # if(len(w)>1):
                #     # there's a tie
                #     lex_neutral = 0
                    
                #     if(singleton_exists(votes)):
                #         lex_mpsr_neutral = 1
                #     else:
                #         lex_mpsr_neutral = 0
                        
                # vals.append([file, m, n, 'NEU', voting_rule.__name__, 'lexicographic_tiebreaking',
                #                 lex_neutral])
                # vals.append([file, m, n, 'NEU', voting_rule.__name__, 'singleton_lex_tiebreaking',
                #                 lex_mpsr_neutral])
                
                df0 = pd.DataFrame(vals, columns = ['file','m','n','axiom','voting_rule',
                                                    'tiebreaking','sat'])
                # print("NEU done")
                                        
            tok = time()
            print("The whole thing took: ", tok - tik)
            
            df = df.append(df0)
            
    df.to_csv(f'axiom_sat-{time_str}.csv', index=False)
