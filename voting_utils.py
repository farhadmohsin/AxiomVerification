import numpy as np

def Copeland_winner(votes):
    """
    Description:
        Calculate Copeland winner given a preference profile
    Parameters:
        votes:  preference profile with n voters and m alternatives
    Output:
        winner: Copeland winner
        scores: pairwise-wins for each alternative
    """
    n,m = votes.shape
    scores = np.zeros(m)
    for m1 in range(m):
        for m2 in range(m1+1,m):
            m1prefm2 = 0        #m1prefm2 would hold #voters with m1 \pref m2
            for v in votes:
                if(v.tolist().index(m1) < v.tolist().index(m2)):
                    m1prefm2 += 1
            m2prefm1 = n - m1prefm2
            if(m1prefm2 == m2prefm1):
                scores[m1] += 0.5
                scores[m2] += 0.5
            elif(m1prefm2 > m2prefm1):
                scores[m1] += 1
            else:
                scores[m2] += 1
    winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
    
    return winner, scores

def maximin_winner(votes):
    """
    Description:
        Calculate maximin winner given a preference profile
    Parameters:
        votes:  preference profile with n voters and m alternatives
    Output:
        winner: maximin winner
        scores: min{D_p(c,c') |c' != c}
    """
    n,m = votes.shape
    Dp_matrix = np.zeros([m,m])
#    scores = np.zeros(m)
    for m1 in range(m):
        for m2 in range(m1+1,m):
            m1prefm2 = 0        #m1prefm2 would hold #voters with m1 \pref m2
            for v in votes:
                if(v.tolist().index(m1) < v.tolist().index(m2)):
                    m1prefm2 += 1
            m2prefm1 = n - m1prefm2
            Dp_matrix[m1][m2] = m1prefm2 - m2prefm1
            Dp_matrix[m2][m1] = m2prefm1 - m1prefm2
    # print(Dp_matrix)        
    scores = np.min(Dp_matrix, axis = 1)
            
    winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
    
    return winner, scores