from preflib_utils import read_preflib_soc
from voting_utils import *
import numpy as np
import os

# def lexicographic_tiebreaking(winners):
#     return winners[0]

naive_dfs_witness_is_found = False

def naive_dfs(depth, k, votes, mask, b, a, r, tiebreaking):
    '''
    Description:
        Check whether the preference profile satisfies GP-r-k or not
        NOTE: stack overflow will happen if n or k is very large
    Parameters:
        depth : the current index of votes array to be removed potentially
        votes: preference profile with n voters and m alternatives
        mask: the masked array of votes array
        b : the target winner
        a : the original winner
        k : the number of votes need to be removed
        r : the voting rule applied
    Output:
        None, but I use global variable (naive_dfs_witness_is_found) here
    '''
    global naive_dfs_witness_is_found
    if k == 0:
        c, tmp_score = r(votes[mask])
        if tiebreaking(votes[mask],c) == b:
            naive_dfs_witness_is_found = True
        return
    if depth == len(votes):
        return

    if votes[depth].tolist().index(b) < votes[depth].tolist().index(a):
        mask[depth] = False
        naive_dfs(depth + 1, k - 1, votes, mask, b, a, r, tiebreaking)
        if naive_dfs_witness_is_found is True:
            # print(mask)
            return
        mask[depth] = True
    naive_dfs(depth + 1, k, votes, mask, b, a, r, tiebreaking)

def brute_force(m, n, n_votes, n_unique, votes, anon_votes, r, tiebreaking, verbose=True):
    global naive_dfs_witness_is_found
    a, score = r(votes)
    a = tiebreaking(votes, a)
    for k in range(1, n):
        for b in range(m):
            if b == a:
                continue
            mask = np.ones(votes.shape[0], dtype = np.bool)
            naive_dfs_witness_is_found = False
            naive_dfs(0, k, votes, mask, b, a, r, tiebreaking)

            if naive_dfs_witness_is_found is True:
                if(verbose):
                    print("GP not satisfied for {} and {}! k = {} and b = {}".format(r.__name__, tiebreaking.__name__,k, b))
                break
        if naive_dfs_witness_is_found is True:
            break
    return naive_dfs_witness_is_found

def main():
    cnt = 0
    tot = 0
    for root, dirs, files in os.walk("./dataset/"):
        for file in files:
            if "ED-00004" in file:
                continue
            tot += 1
            m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc("./dataset/" + file)
            assert n == n_votes
            print(file)
            if brute_force(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner,
                  lexicographic_tiebreaking) is True:
                cnt += 1
    print(cnt, tot, cnt / tot)

if __name__ == "__main__":
    main()