import pandas as pd
import numpy as np
import cvxpy as cp

from voting_rules import Copeland_winner, maximin_winner

def read_preflib_soc(filename):
    '''

    Parameters
    ----------
    filename : STR
        soc file to read

    Returns
    -------
    m : no_candidates
    n : no_voters
    n_votes : no_votes (might be different from n, but is usually the same)
    n_unique : number of uniqu ranking that's present, maximum m!
    pref_profile: list of n rankings
    anon_pref_profile : list of n_unique (ranking, count) pair
    '''
    
    with open(filename) as f:
        txt = f.read()
    lines = txt.split('\n')
    m = int(lines[0])
    n, n_votes, n_unique = [int(x) for x in lines[m+1].split(',')]
    
    pref_profile = []
    anon_pref_profile = []
    for i in range(m+2, len(lines)):
        if(not(lines[i])):
            continue
        row = [int(x) for x in lines[i].split(',')]
        
        anon_pref_profile.append([row[0], np.array(row[1:])-1])
        
        for j in range(row[0]):
            pref_profile.append(np.array(row[1:])-1)
    
    return m, n, n_votes, n_unique, np.array(pref_profile), anon_pref_profile
        
# %% example
fil = 'C:/RPI/CompSoc/Voting Axiom Verification/dataset/ED-00009-00000001.soc'
m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc(fil)
