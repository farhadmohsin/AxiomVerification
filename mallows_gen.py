import numpy as np
from itertools import permutations

def sample_mallows(W, phi, verbose = False):
    """
    Parameters
    ----------
    W : ranking
    phi : distance parameter, 0 < phi <= 1
        low phi means more rankings closer to W
        high phi means more random rankings

    Returns
    -------
    A sample ranking drawn from Mallows(W, phi)
    """
    
    R = [W[0]] # initialize ranking with W[0]
    
    m = len(W)
    for i in range(2, m+1):
        a = W[i-1]
        Z = (1 - phi**i)/(1-phi)
        pi = np.array([phi**(i-k) for k in range(1,i+1)])
        pi = pi/Z
        if(verbose):
            print(f'pos {i} has {a}, pi = {pi}')

        pos = np.random.choice(np.arange(i), p = pi)
        if(verbose):
            print(f'pos={pos}')
        R.insert(pos, a)
    
    if(verbose):
        print(f'sampled ranking is {R}')
    return R

def gen_mallows_profile(N, W, phi):
    votes = []
    for i in range(N):
        votes.append(sample_mallows(W, phi))
    return np.array(votes)

#%%

if __name__ == '__main__':
    
    m = 4
    A = np.arange(m)
    perms = [list(P) for P in list(permutations(A))]
    
    W = np.arange(m)
    phi = 0.7
    
    # show the distribution of preference profiles
    cnt = np.zeros(np.math.factorial(m))
    for t in range(1000):
        R = sample_mallows(W, phi)
        for i,P in enumerate(perms):
            if R==P:
                cnt[i] += 1
    
    print(list(zip(perms,cnt)))
    
    