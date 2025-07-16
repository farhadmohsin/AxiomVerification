from voting_rules import *
from voting_utils import majority_graph
from various_profiles_gen import *
from matplotlib import pyplot as plt

def inconsistent_examples(voting_rule, N=250, m=6, profiles=1000, gen_method = gen_pref_profile):
    votes_all = []
    points_all = []

    tic = time()
    for t in range(profiles):
        # votes = gen_method(N, m)
        # votes_all.append(votes)
        # overriding
        g, candidates = Gaussian_voters_candidates(N, m, 2)
        votes = random_pref_profile(g, candidates)
        votes_all.append(votes)
        points_all.append(dict(voters=g, candidates=candidates))
    toc = time()
    print("Profiles generated in %lf s"%(toc-tic))
    
    # calculate all winners given some voting rule and tiebreaking scheming
    winner_sets = [[] for i in range(m)]
    tic = time()
    for t in range(profiles):
        winner, _ = voting_rule(votes_all[t])
        w = lexicographic_tiebreaking(votes_all[t], winner)
        winner_sets[w].append(t)
    toc = time()
    print("Winners computed in %lf s"%(toc-tic))
    
    examples = []
    points = []
    total_pair_cnt = 0
    
    tic = time()

    for w,wset in enumerate(winner_sets):
        len_set = len(wset)
        total_pair_cnt += (len_set*(len_set - 1))/2
        for i1 in range(len_set):
            for i2 in range(i1+1,len_set):
                joined = np.append(votes_all[wset[i1]],votes_all[wset[i2]],axis = 0)
                winner, _ = voting_rule(joined)
                w_new = lexicographic_tiebreaking(joined, winner)
                if(w_new != w):
                    examples.append([votes_all[wset[i1]], votes_all[wset[i2]], joined])
                    points.append([points_all[wset[i1]], points_all[wset[i2]]])

    toc = time()

    print(f'{len(examples)} profile inconsistencies out of {total_pair_cnt} pairs')            

    return examples, points

if __name__ == '__main__':
    # Copeland_examples = inconsistent_examples(Copeland_winner, 101, 4, 1000, uniform_euclid_profile)
    Copeland_examples, points = inconsistent_examples(Copeland_winner, 101, 4, 10000, gaussian_profile)