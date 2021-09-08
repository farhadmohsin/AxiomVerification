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

voting_rule = Copeland_winner
tiebreaking = singleton_lex_tiebreaking

#%%
file = 'ED-00009-00000001.soc'
m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc("./dataset/" + file)

exist, cond_satisfaction = Condorcet_sat(votes, voting_rule, tiebreaking)
#%%
voting_rule = veto_winner
tiebreaking = lexicographic_tiebreaking
file = 'ED-00011-00000001.soc'
m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc("./dataset/" + file)

exist, cond_satisfaction = Condorcet_sat(votes, voting_rule, tiebreaking)
