from various_profiles_gen import *
from ilp_gp_gurobipy import ILP_Copeland_lexicographic_new, ILP_maximin_lexicographic_new, \
                            ILP_Blacks_lexicographic_new_2, ILP_STV_lexicographic
from voting_utils import *
from mallows_gen import *
from pref_profile_gen import *
from time import time
from datetime import datetime
from tqdm import tqdm
from pprint import pprint
import copy
import sys
from itertools import combinations, permutations

def fprint(filename, *texts):
    # Open the file in append mode (or create if it doesn't exist)
    with open(filename, 'a') as file:
        # Append each text to the file
        text = ' '.join(str(t) for t in texts)
        file.write(text)
        # Add a newline character for better readability
        file.write('\n')

def test(*argv):
    
    '''
    first argument is test_type
    for Mallows and PL, followed by second argument
    '''
    time_now = datetime.now().strftime('%y%m%d%H%M%S')
    
    print(argv)
        
    # m = 4
    n = 100
    
    rule = argv[0]
    
    if rule == 'Copeland':
        test_GNSP = ILP_Copeland_lexicographic_new
    elif rule == 'Blacks':
        test_GNSP = ILP_Blacks_lexicographic_new_2
    elif rule == 'maximin':
        test_GNSP = ILP_maximin_lexicographic_new
    elif rule == 'STV':
        test_GNSP = ILP_STV_lexicographic
        
    distribution = argv[1]
    
    np.random.seed(0)
    
    test = []
    test_ILP = []
    ILP_soln_all = []
    test_ILP_new = []
    ILP_new_soln_all = []

    # n_all = list(range(10, 100, 10)) + list(range(100, 1001, 100))  
    # n_all = list(range(10, 101, 10))
    
    m_all = list(range(3, 7, 1))
    
    # for n in n_all:
    for m in m_all:
        
        if rule == 'STV':
            perms = [list(p) for p in permutations(list(range(m)))]
        # print(n)    
    
        ILP_cnt = 0
        
        ILP_new_succ_times = []
        ILP_new_fail_times = []
        ILP_new_soln = []
        
        Votes = []
        samples = 100
        
        # IC
        if(distribution == 'IC'):
            for s in tqdm(range(samples)):
                votes = gen_pref_profile(n, m)
                Votes.append(votes)
        
        # Mallows
        elif(distribution == 'Mallows'):
            W = np.array(range(m))
            phi = float(argv[2])
            for s in tqdm(range(samples)):
                votes = gen_mallows_profile(n, W, phi)
                Votes.append(votes)
             
        # PL
        elif(distribution == 'PL'):
            PL_type = argv[2]
            gamma = create_PL_params(m)
            if PL_type == 'top-2':
                t = 0.5 * (gamma[0] + gamma[1])
                gamma[0] = t
                gamma[1] = t
                
            for s in tqdm(range(samples)):
                votes = gen_PL_ballot(gamma, n, m)
                Votes.append(votes)
        
        # urns
        elif(distribution == 'Urns'):
            alpha = 0.01
            for s in tqdm(range(samples)):
                votes = sample_urns_profile(n, m, alpha = alpha)
                Votes.append(votes)
        
        # single-peaked
        elif(distribution == 'single-peaked'):
            for s in tqdm(range(samples)):
                votes = single_peaked_profile(n, m)
                Votes.append(votes)
                
        # uniform-euclidean
        elif(distribution == 'unif-euclidean'):
            for s in tqdm(range(samples)):
                votes = euclidean_profile(n, m, 'U')
                Votes.append(votes)
        
        # Gaussian-euclidean
        elif(distribution == 'Gauss-euclidean'):
            for s in tqdm(range(samples)):
                votes = euclidean_profile(n, m, 'G')
                Votes.append(votes)
            
        
        for s in range(samples):
            # print(s)
            votes = Votes[s]
            m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
            
            # ILP_new_Copeland
            tik = time()
            # flag_ILP_new, k, new_votes, lp_flag = ILP_Copeland_lexicographic_new(votes, anon_votes,
            #                                                           n, m, debug = False)
            if (rule == 'STV'):
                flag_ILP_new, k, lp_flag = ILP_STV_lexicographic(votes, anon_votes, n, m, 
                                                          perms, debug = False)
                new_votes = None
            
            else:
                flag_ILP_new, k, new_votes, lp_flag = test_GNSP(votes, anon_votes,
                                                                n, m, debug = False)
                
            tok = time()
            print(f'{(tok - tik)=}')
            if(flag_ILP_new):
                ILP_cnt += 1
                pp = copy.deepcopy(votes)
                anon_pp = copy.deepcopy(anon_votes)
                ILP_new_succ_times.append(tok-tik)
            else:
                ILP_new_fail_times.append(tok-tik)
            ILP_new_soln.append([flag_ILP_new, k, new_votes, lp_flag])                        
            
            
        print('===============================')
        print(f'{n=}')
        
        print('ILP_new', ILP_cnt, np.mean(ILP_new_succ_times), np.mean(ILP_new_fail_times))
        # test_ILP.append([n, ILP_cnt, ILP_new_succ_times, ILP_new_fail_times])
        test_ILP.append([m, ILP_cnt, ILP_new_succ_times, ILP_new_fail_times])
        ILP_soln_all.append(ILP_new_soln)

        print('===============================')
    
    if(distribution == 'Mallows'):
        distribution_print = f'Mallows_{phi}'
    elif(distribution == 'PL'):
        distribution_print = f'PL_{PL_type}'
    else:
        distribution_print = distribution
    
    print(distribution_print)
    
    # filename = f'{time_now}_{rule}_ILP_GP_{m=}_{distribution_print=}_{samples=}'
    filename = f'{time_now}_{rule}_ILP_GP_{n=}_{distribution_print=}_{samples=}'
    
    for output in test_ILP:
        outfile = f'{filename}.out'
        fprint(outfile, '===============================')
        # fprint(outfile, f'n={output[0]}')
        fprint(outfile, f'm={output[0]}')
        
        fprint(outfile, 'ILP_new', output[1], np.mean(output[2] + output[3]))
        fprint(outfile, '===============================')
        
    npyfile = f'{filename}.npy'
    with open(npyfile, 'wb') as f:
        np.save(f, np.array(test_ILP, dtype = object))
        np.save(f, np.array(ILP_soln_all, dtype = object))
    
    return test_ILP
    
if __name__ == '__main__':
    argv = sys.argv[1:]
    test_ILP = test(*argv)
    
    # have to do the fixed n test again
    # n = 100?