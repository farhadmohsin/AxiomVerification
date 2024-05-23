import numpy as np

def draw_pl_vote(m, gamma):
    """
    Description:
        Generate a Plackett-Luce vote given the model parameters.
    Parameters:
        m:     number of alternatives
        gamma: parameters of the Plackett-Luce model
    """

    localgamma = np.copy(gamma) # work on a copy of gamma
    localalts = np.arange(m) # enumeration of the candidates
    vote = []
    for j in range(m): # generate position in vote for every alternative

        # transform local gamma into intervals up to 1.0
        localgammaintervals = np.copy(localgamma)
        prev = 0.0
        for k in range(len(localgammaintervals)):
            localgammaintervals[k] += prev
            prev = localgammaintervals[k]

        selection = np.random.random() # pick random number

        # selection will fall into a gamma interval
        for l in range(len(localgammaintervals)): # determine position
            if selection <= localgammaintervals[l]:
                vote.append(localalts[l])
                localgamma = np.delete(localgamma, l) # remove that gamma
                localalts = np.delete(localalts, l) # remove the alternative
                localgamma /= np.sum(localgamma) # renormalize
                break
    return vote

def create_PL_params(m):
    # gamma = np.random.uniform(size = m)
    gamma = np.random.normal(size = m)
    gamma = np.exp(gamma)
    gamma /= np.sum(gamma)
    return gamma

def gen_PL_ballot(gamma, n, m):
    ballots = []
    for i in range(n):
        ballots.append(draw_pl_vote(m, gamma))
    return np.array(ballots)

if __name__ == '__main__':
    m = 4
    gamma = create_PL_params(m)
    votes = gen_PL_ballot(gamma, 30, m)