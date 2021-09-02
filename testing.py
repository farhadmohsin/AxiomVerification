import numpy as np
from preflib_utils import read_preflib_soc
from voting_utils import *

# %% sampling preference profiles


samples = 1000 # no_of_samples
np.random.seed(0)

N = 100
m = 4
sample_pp = [gen_pref_profile(N, m) for s in range(samples)]
sample_pp = np.array(sample_pp)

# %% Varius satisfaction functions using sampling

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
        if(winner == v_w0):
            cond_satisfaction = 1
    return exist, cond_satisfaction

# example, for Borda winner and lexicographic tiebreaking
#   TODO: eventually loop through voting rules and tiebreaking mechanisms

# Condorcet criterion satisfaction

def Condorcet_saitsfaction(sample_pp):

    cnt_exist = 0
    cnt_sat = 0
    for votes in sample_pp:
        cond_exist, cond_satisfaction = Condorcet_sat(votes, Borda_winner, lexicographic_tiebreaking)
        
        cnt_exist += cond_exist
        cnt_sat += cond_satisfaction
    
    return (cnt_exist, cnt_sat)

cnt_exist, cnt_sat = Condorcet_satisfaction(sample_pp)
print(cnt_sat/cnt_exist)