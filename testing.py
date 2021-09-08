import numpy as np
from preflib_utils import read_preflib_soc
from voting_utils import *
from bruteforce import *
from time import time
from datetime import datetime
from tqdm import tqdm

# %% Varius satisfaction functions using sampling

#%% Condorcet satisfaction

''' condorcet_exists from voting_utils checks if Condorcet winner exists'''

'''checking for anonymity and neutrality boils down to checking for ties and
        which tiebreaking function is used
    - the tiebreaking functions are 
        * lexicographic_tiebreaking - anonymous
        * voter1_tiebreaking - neutral
        * two forms of most popular singleton tiebreaking
            * singleton_lex_tiebreaking - anonymous
            * singleton_v1_tiebreaking - neutral
'''

def Condorcet_sat(votes, voting_rule, tiebreaking):
    """
    
    Returns
    -------
    returns condorcet_exists, condorcet_satisfaction

    """
    w, s = voting_rule(votes)
    exist, winner = condorcet_exist(votes)
    cond_satisfaction = 0
    if(exist):
        v_winners, _ = voting_rule(votes)
        v_w0 = tiebreaking(votes, v_winners)
        if(winner[0] == v_w0):
            cond_satisfaction = 1
    return exist, cond_satisfaction

# example, for Borda winner and lexicographic tiebreaking
#   TODO: eventually loop through voting rules and tiebreaking mechanisms

# Condorcet criterion satisfaction

def Condorcet_saitsfaction(sample_votes, voting_rule, tiebreaking):

    cnt_exist = 0
    cnt_sat = 0
    samples = len(sample_votes)
    for votes in sample_votes:
        cond_exist, cond_satisfaction = Condorcet_sat(votes, voting_rule, tiebreaking)
        
        cnt_exist += cond_exist
        cnt_sat += cond_satisfaction
    
    return cnt_exist, cnt_sat, (cnt_sat+samples-cnt_exist)/samples

# %% Neutrality and Anonymity


def singleton_exists(votes):
    
    # changing this for when m is large
    
    # old 
    # rank_cnt = ranking_count(votes)
    # rank_srt = np.flip(np.argsort(rank_cnt))
    
    # new
    m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
    rank_cnt = [av[0] for av in anon_votes]
    rank_srt = np.flip(np.argsort(rank_cnt))
    
    flag = False
    for r in rank_srt:
        rank = rank_cnt[r]
        l = len(np.argwhere(rank_cnt==rank))
        if(l==1):
            flag = True
            break
    
    return flag
        

def neutrality_anonymity_satisfaction(sample_votes, voting_rule):
    
    # we just check if there's a tie or not
    #   no need to check tiebreaking rule here, because we can interpret it theoretically
    #   if tie, lex not not neutral, voter1 not anonymous
    #   then check if most popular singleton exists
    #   if does not, singleton lex not neutral, singleton voter1 not anonymous
    
    lex_neutral_cnt = 0
    v1_anon_cnt = 0
    lex_mpsr_neutral_cnt = 0
    v1_mpsr_anon_cnt = 0
    
    samples = len(sample_votes)
    
    for votes in sample_votes:
        w, s = voting_rule(votes)
        if(len(w)>1):
            # there's a tie
            lex_neutral_cnt += 1
            v1_anon_cnt += 1
            
            if(singleton_exists(votes)):
                lex_mpsr_neutral_cnt += 1
                v1_mpsr_anon_cnt += 1
                
    return lex_neutral_cnt, v1_anon_cnt, lex_mpsr_neutral_cnt, v1_mpsr_anon_cnt

#%% Group pariticipation

def participation_sat(sample_votes, voting_rule, tiebreaking):
    
    val = []
    for i, votes in enumerate(tqdm(sample_votes)):
        tik = time()
        
        # return run-time, and satisfaction value
        m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
        if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, voting_rule, tiebreaking, verbose = False)):
            flag = 1
        else:
            flag = 0
           
        tok = time()
        val.append([flag, tok-tik])
        
    return val
    

if __name__ == "__main__":
    
    
    vals = []
    voting_rules = [plurality_winner, Borda_winner, veto_winner, 
                    Copeland_winner, maximin_winner]
    tiebreaking_methods = [lexicographic_tiebreaking, voter1_tiebreaking,
                            singleton_lex_tiebreaking, singleton_v1_tiebreaking]
    # voting_rules = [Copeland_winner]
    # tiebreaking_methods = [singleton_lex_tiebreaking]
    
    tie_pairs = list(zip(['lexicographic_tiebreaking', 'voter1_tiebreaking',
                           'singleton_lex_tiebreaking', 'singleton_v1_tiebreaking'],
                         ['Neutrality', 'Anonymity', 'Neutrality', 'Anonymity']))
    
    #  sampling preference profiles
    
    samples = 1000 # no_of_samples
    np.random.seed(0)
    
    m = 4
    
    for N in range(20, 50, 5):
        
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(N, m, samples)
        
        tik = time()
        
        # save  the preference profile
        sample_votes = [gen_pref_profile(N, m) for s in range(samples)]
        sample_votes = np.array(sample_votes)
        
        # testing
        
        # TODO: Uncomment everything else for the rest
        #   right now, only checking group participation
        for voting_rule in voting_rules:
            for tiebreaking in tiebreaking_methods:
                
                # Condorcet satisfaction for various tiebreaking
                cnt_exist, cnt_sat, cond_sat = Condorcet_saitsfaction(sample_votes, 
                                                                      voting_rule, tiebreaking)
                cond_sat = cnt_sat+samples-cnt_exist
                
                vals.append([N, m, samples, voting_rule.__name__, tiebreaking.__name__,
                            'Condorcet', cond_sat])
                
                # taking out group participation
                # group participation
                # participation_vals = participation_sat(sample_votes, voting_rule, tiebreaking)
                # vals.append([N, m, samples, voting_rule.__name__, tiebreaking.__name__,
                            # 'Group Participation', samples - participation_cnt])
                
                # save the output vals here
                
            
            # Neutrality, anonimity for relevant tiebreaking
            lex_neutral_cnt, v1_anon_cnt, lex_mpsr_neutral_cnt, v1_mpsr_anon_cnt = neutrality_anonymity_satisfaction(sample_votes, voting_rule)
            cnts = [lex_neutral_cnt, v1_anon_cnt, lex_mpsr_neutral_cnt, v1_mpsr_anon_cnt]
            for i, cnts in enumerate(cnts):
                vals.append([N, m, samples, voting_rule.__name__, tie_pairs[i][0],
                                tie_pairs[i][1], samples - cnts])
                
            with open(f'output-{N}-{voting_rule.__name__}-{time_str}.npy', 'wb') as f:
                np.save(f, vals)
        
                
        tok = time()
        print("The whole thing took: ", tok - tik)