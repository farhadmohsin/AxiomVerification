from preflib_utils import read_preflib_soc
from voting_utils import *
import numpy as np
import os
from itertools import permutations

def lexicographic_tiebreaking(winners):
	return winners[0]

def Copeland_winner_anon(anon_votes, removed_cnt = None):
	"""
	Description:
		Calculate Copeland winner given a preference profile
	Parameters:
		anon_votes:  preference profile with n voters and m alternatives and the number of votes
		removed_cnt: the array of the number of removed votes from anon_votes
	Output:
		winner: Copeland winner
		scores: pairwise-wins for each alternative
	"""
	n = len(anon_votes)
	m = anon_votes[0][1].shape[0]
	scores = np.zeros(m)
	for m1 in range(m):
		for m2 in range(m1+1,m):
			m1prefm2 = 0		#m1prefm2 would hold #voters with m1 \pref m2
			m2prefm1 = 0
			for i, anon_vote in enumerate(anon_votes):
				cnt1, v = anon_vote
				if removed_cnt is None:
					cnt2 = 0
				else:
					cnt2 = removed_cnt[i]
				if v.tolist().index(m1) < v.tolist().index(m2):
					m1prefm2 += cnt1 - cnt2
				else:
					m2prefm1 += cnt1 - cnt2
			if m1prefm2 == m2prefm1:
				scores[m1] += 0.5
				scores[m2] += 0.5
			elif m1prefm2 > m2prefm1:
				scores[m1] += 1
			else:
				scores[m2] += 1
	winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
	return winner, scores

naive_dfs_witness_is_found = False

def naive_dfs(depth, anon_votes, removed_cnt, b, a, r, tiebreaking, verbose = False):
	global naive_dfs_witness_is_found
	if depth == len(anon_votes):
		c, tmp_score = r(anon_votes, removed_cnt)
		# if verbose:
		# 	print(b, c, removed_cnt)
		if tiebreaking(c) == b:
			naive_dfs_witness_is_found = True
		return

	cnt1, vote = anon_votes[depth]
	if vote.tolist().index(b) < vote.tolist().index(a):
		for cnt2 in range(1, cnt1 + 1):
			removed_cnt[depth] = cnt2
			naive_dfs(depth + 1, anon_votes, removed_cnt, b, a, r, tiebreaking, verbose)
			if naive_dfs_witness_is_found is True:
				return
			removed_cnt[depth] = 0
	naive_dfs(depth + 1, anon_votes, removed_cnt, b, a, r, tiebreaking, verbose)

def algo(m, n, n_votes, n_unique, votes, anon_votes, r, tiebreaking, verbose = False):
	global naive_dfs_witness_is_found
	a, score = r(anon_votes)
	a = tiebreaking(a)
	for b in range(m):
		if b == a:
			continue
		removed_cnt = np.zeros(len(anon_votes), dtype = np.int)
		naive_dfs_witness_is_found = False
		naive_dfs(0, anon_votes, removed_cnt, b, a, r, tiebreaking, verbose = verbose)

		if naive_dfs_witness_is_found is True:
			if verbose:
				k = np.sum(removed_cnt)
				print("GP not satisfied for {} and {}! k = {} and b = {}".format(r.__name__, tiebreaking.__name__, k, b))
				print("The numbers of each removed vote are {}".format(removed_cnt))
			break
	return naive_dfs_witness_is_found

def main(verbose = False):
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
			if algo(m, n, n_votes, n_unique, votes, anon_votes, Copeland_winner_anon, lexicographic_tiebreaking, verbose = verbose) is True:
				cnt += 1
	print(cnt, tot, cnt / tot)

if __name__ == "__main__":
	main(verbose = True)