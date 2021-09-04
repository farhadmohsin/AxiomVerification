from preflib_utils import read_preflib_soc
from voting_utils import Copeland_winner, maximin_winner
import numpy as np
import os
import itertools
import math

def lexicographic_tiebreaking(winners):
	return winners[0]

AI_copeland_witness_is_found = False

def naive_dfs(depth, k, votes, mask, b, a):
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
	Output:
		None, but I use global variable (AI_copeland_witness_is_found) here
	'''
	global AI_copeland_witness_is_found
	if k == 0:
		c, tmp_score = Copeland_winner(votes[mask])
		if lexicographic_tiebreaking(c) == b:
			AI_copeland_witness_is_found = True
		return
	if depth == len(votes):
		return

	if votes[depth].tolist().index(b) < votes[depth].tolist().index(a):
		mask[depth] = False
		naive_dfs(depth + 1, k - 1, votes, mask, b, a)
		if AI_copeland_witness_is_found is True:
			return
		mask[depth] = True
	naive_dfs(depth + 1, k, votes, mask, b, a)

def AppMoVCopeland(d, m, n, n_votes, n_unique, votes, anon_votes, alpha = 0.5):
	"""
	Returns an integer that is equal to the margin of victory of the election profile, that is,
	the smallest number k such that changing k votes can change the winners.
	"""
	wmgMap = np.zeros((m, m), dtype = np.int32)
	for m1 in range(m):
		for m2 in range(m1 + 1, m):
			m1prefm2 = 0
			for v in votes:
				if v.tolist().index(m1) < v.tolist().index(m2):
					m1prefm2 += 1
			m2prefm1 = n - m1prefm2
			wmgMap[m1][m2] = m1prefm2 - m2prefm1
			wmgMap[m2][m1] = m2prefm1 - m1prefm2

	# Compute c* = argmin_c RM(d,c)
	relative_margin = {}
	for c in range(m):
		if c == d:
			continue
		relative_margin[c] = RM(wmgMap, n, m, d, c, alpha)
	c_star = min(relative_margin.items(), key = lambda x: x[1])[0]
	return relative_margin[c_star] * (math.ceil(np.log(m)) + 1)

def RM(wmgMap, n, m, d, c, alpha = 0.5):
	for t in range(n):
		# Compute s_-t_d and s_t_c
		s_neg_t_d = 0
		s_t_c = 0
		for e in range(m):
			if e == d:
				continue
			if wmgMap[e][d] < -2 * t:
				s_neg_t_d += 1.0
			elif wmgMap[e][d] == -2 * t:
				s_neg_t_d += alpha
		for e in range(m):
			if e == c:
				continue
			if wmgMap[e][c] < 2 * t:
				s_t_c += 1.0
			elif wmgMap[e][c] == 2 * t:
				s_t_c += alpha

		if s_neg_t_d <= s_t_c:
			return t

def AI_copeland(m, n, n_votes, n_unique, votes, anon_votes):
	global AI_copeland_witness_is_found
	a, score = Copeland_winner(votes)
	a = lexicographic_tiebreaking(a)

	mov_k = AppMoVCopeland(a, m, n, n_votes, n_unique, votes, anon_votes)
	print(mov_k, n)
	for k in range(mov_k, n):
		for b in range(m):
			if b == a:
				continue
			mask = np.ones(votes.shape[0], dtype = np.bool)
			AI_copeland_witness_is_found = False
			naive_dfs(0, k, votes, mask, b, a)

			if AI_copeland_witness_is_found is True:
				print("GP not satisfies! k = {} and b = {}".format(k, b))
				break
		if AI_copeland_witness_is_found is True:
			break
	return AI_copeland_witness_is_found

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
			if AI_copeland(m, n, n_votes, n_unique, votes, anon_votes) is True:
				cnt += 1
	print(cnt, tot, cnt / tot)

if __name__ == "__main__":
	main()
