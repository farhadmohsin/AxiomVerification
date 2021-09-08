import numpy as np
from voting_utils import *
from bruteforce import *
from itertools import combinations
from scipy.optimize import linprog
import copy
from time import time
from ilp_gp import *


if __name__ == '__main__':
    cnt = 0
    tot = 0
    for root, dirs, files in os.walk("./dataset/"):
        for file in files:
            if not "ED-00012" in file:
                continue
            tot += 1
            m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc("./dataset/" + file)
            assert n == n_votes
            print(file)
#             if algo(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner_anon, lexicographic_tiebreaking, verbose = verbose) is True:
#                 cnt += 1
            
            voting_rule = Copeland_winner
            tiebreaking = lexicographic_tiebreaking
            
            # participation_sat = ILP_search(votes, anon_votes, n, m, voting_rule, tiebreaking, debug=True)
            try:
                participation_sat = ILP_search(votes, anon_votes, n, m, voting_rule, 
                                               tiebreaking, debug = True)
                if(participation_sat):
                    cnt += 1
            except Exception:
                print(file, "could not be worked with")
