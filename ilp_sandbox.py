from ilp_gp_new import *

if __name__ == '__main__':
    
    time_now = datetime.now().strftime('%y%m%d%H%M%S')
    
    n = 20
    m = 4
    distribution = 'IC'
    
    np.random.seed(0)
    
    edges, UMGS = create_umgs(m)
    print('UMGs created')
    
    # maximin_pairs = product_alt_pairs(m)
    # print('maximin pairs created', len(maximin_pairs))
    
    perms = [list(p) for p in permutations(list(range(m)))]
    print('permutations created')
    
    test = []
    test_ILP = []
    ILP_soln_all = []
    
    brute_cnt = 0
    ILP_cnt = 0
    
    brute_times = []
    ILP_times = []
    ILP_soln = []
    
    Votes = []
    samples = 100
    
    # IC
    if(distribution == 'IC'):
        for s in range(samples):
            votes = gen_pref_profile(n, m)
            Votes.append(votes)
    # Mallows
    elif(distribution == 'Mallows'):
        W = np.array(range(m))
        phi = 0.5
        for s in range(samples):
            votes = gen_mallows_profile(n, W, phi)
            Votes.append(votes)
    
    brute_cnt = 0
    ILP_cnt = 0
    
    for s in range(samples):
        # print(s)
        votes = Votes[s]
        m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
        # anon_votes = np.array(anon_votes)
        
        brute_flag = brute_force(m, n, n_votes, n_unique, votes, anon_votes, Blacks_winner, 
                            lexicographic_tiebreaking)
        flag, k, lp_flag = ILP_Blacks_lexicographic(votes, anon_votes,
                                                    n, m, UMGS, edges, debug = False)
        
        print(s, brute_flag, flag)
        
        brute_cnt += 1 if brute_flag else 0
        ILP_cnt += flag
        
        # if(brute_flag or flag == 1):
        #     break
        
        
    print(brute_cnt, ILP_cnt)