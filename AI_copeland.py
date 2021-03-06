from preflib_utils import read_preflib_soc
from voting_utils import Copeland_winner, maximin_winner, lexicographic_tiebreaking
import numpy as np
import os
import itertools
import math
import copy
from itertools import combinations
from ilp_gp import *
import time

# def lexicographic_tiebreaking(winners):
# 	return winners[0]

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
		for m2 in range(m1 + 1, m):
			m1prefm2 = 0  # m1prefm2 would hold #voters with m1 \pref m2
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

LP_times = []
LP_total = 0
LP_cnt = 0

def LP_search(anon_votes, removed_cnt, b, a, tiebreaking, debug = False):
	global LP_times, LP_total, LP_cnt
	LP_total += 1
	tic = time.time()

	tmp_anon_votes = copy.deepcopy(anon_votes)
	for i in range(len(tmp_anon_votes)):
		# cnt, v = tmp_anon_votes[i]
		tmp_anon_votes[i][0] -= removed_cnt[i]

	n = len(tmp_anon_votes)
	m = tmp_anon_votes[0][1].shape[0]
	# initializtion
	w_, s = Copeland_winner_anon(tmp_anon_votes, removed_cnt)
	w = tiebreaking(None, w_)

	rankings = [av[1] for av in tmp_anon_votes]

	# pre-compute combinations
	alts = np.arange(m - 1)
	all_combinations = []
	for i in range((m + 1) // 2, m):
		all_combinations.append(combs(alts, i))

	vbw = Rab(b, w, rankings)  # ranking indices with b \succ w
	vwb = Rab(w, b, rankings)  # ranking indices with w \succ b

	# found_flag = False
	if len(vbw) == 0:
		toc = time.time()
		LP_times.append(toc - tic)
		return False
	for i in range((m + 1) // 2, m):
		# if(found_flag):
		#     break
		# for different no. head-to-head wins
		i_combs = all_combinations[i - (m + 1) // 2]

		for temp in i_combs:
			# for different combinations of wins
			C = temp.copy()
			C[C == b] = m - 1  # replace
			if debug:
				print(f"b = {b}, C = {C}")

			# create the LP
			# compute A and h (obj. fun. would be 0 since we just want feasibility) Ax <= h
			# len(x) = len(vbw), because these are the only rankings manipulatable
			A, h, c = create_LP(tmp_anon_votes, b, vbw, vwb, C, rankings)

			sol = cvxopt.solvers.lp(c, A.T, h)
			# print(sum(c.T * sol['x']))
			if sol['status'] != 'optimal':
				if debug:
					print('LP not feasible:', sol['status'])
				continue
			else:
				toc = time.time()
				LP_times.append(toc - tic)
				return True

			# status, x = cvxopt.glpk.ilp(c, A.T, h, I=set(range(len(c))))
			# 	# print(sum(c.T*x))
			# if status != 'optimal' :
			# 	if debug :
			# 		print('ILP not feasible', 'status')
			# 	continue
			# else:
			# 	if debug:
			# 		print('ILP feasible')
			# 	toc = time.time()
			# 	LP_times.append(toc - tic)
			# 	return True

	toc = time.time()
	LP_times.append(toc - tic)
	LP_cnt += 1
	return False

def leastVotesToRemove(anon_votes, removed_cnt, b, a, tiebreaking):
	n = len(anon_votes)
	m = anon_votes[0][1].shape[0]

	# a, tmp_score = Copeland_winner_anon(anon_votes, removed_cnt)
	# a = a[0]

	# weights of a in wmg: P[a > c] - P[c > a]
	w_a = np.zeros(m, dtype = np.int32)
	# weights of b im wmg: P[b > c] - P[c > b]
	w_b = np.zeros(m, dtype = np.int32)
	for i, anon_vote in enumerate(anon_votes):
		cnt1, v = anon_vote
		if removed_cnt is None:
			cnt2 = 0
		else:
			cnt2 = removed_cnt[i]
		for c in range(m):
			if a != c:
				if v.tolist().index(a) < v.tolist().index(c):
					w_a[c] += cnt1 - cnt2
				else:
					w_a[c] -= cnt1 - cnt2
			if b != c:
				if v.tolist().index(b) < v.tolist().index(c):
					w_b[c] += cnt1 - cnt2
				else:
					w_b[c] -= cnt1 - cnt2

	min_out_a = -1
	min_in_b = -1
	for c in range(m):
		if a != c:
			if w_a[c] >= 0 and (min_out_a == -1 or w_a[c] < min_out_a):
				min_out_a = w_a[c]
		if b != c:
			if w_b[c] <= 0 and (min_in_b == -1 or -w_b[c] < -min_in_b):
				min_in_b = -w_b[c]

	# print(w_a, min_out_a, w_b, min_in_b)
	return min(min_out_a, min_in_b)

AI_copeland_witness_is_found = False

LV_total = 0
LV_cnt = 0

def naive_dfs(depth, k, anon_votes, removed_cnt, b, a, tiebreaking):
	'''
	Description:
		Check whether the preference profile satisfies GP-r-k or not
		NOTE: stack overflow will happen if n or k is very large
	Parameters:
		depth : the current index of votes array to be removed potentially
		anon_votes: preference profile with n voters and m alternatives and the number of votes
		removed_cnt: the array of the number of removed votes from anon_votes
		b : the target winner
		a : the original winner
		k : the number of votes need to be removed
	Output:
		None, but I use global variable (AI_copeland_witness_is_found) here
	'''
	global AI_copeland_witness_is_found, LV_total, LV_cnt
	if k == 0:
		c, tmp_score = Copeland_winner_anon(anon_votes, removed_cnt)
		if tiebreaking(None, c) == b:
			AI_copeland_witness_is_found = True
		return
	if depth == len(anon_votes):
		return

	# Pruning
	# if not LP_search(anon_votes, removed_cnt, b, a, tiebreaking):
	# 	return
	LV = leastVotesToRemove(anon_votes, removed_cnt, b, a, tiebreaking)
	LV_total += 1
	if LV > k:
		LV_cnt += 1
		return

	cnt1, vote = anon_votes[depth]
	if vote.tolist().index(b) < vote.tolist().index(a):
		for cnt2 in range(1, min(cnt1, k) + 1):
			removed_cnt[depth] = cnt2
			naive_dfs(depth + 1, k - cnt2, anon_votes, removed_cnt, b, a, tiebreaking)
			if AI_copeland_witness_is_found is True:
				return
			removed_cnt[depth] = 0
	naive_dfs(depth + 1, k, anon_votes, removed_cnt, b, a, tiebreaking)

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
	return relative_margin[c_star]

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

def index_of_winner(vote, a):
	for i, x in enumerate(vote):
		if x == a:
			return i

def AI_copeland(m, n, n_votes, n_unique, votes, anon_votes, tiebreaking):
	global AI_copeland_witness_is_found

	scores = np.zeros(m)
	for m1 in range(m):
		for m2 in range(m1 + 1, m):
			m1prefm2 = 0  # m1prefm2 would hold #voters with m1 \pref m2
			for v in votes:
				if v.tolist().index(m1) < v.tolist().index(m2):
					m1prefm2 += 1
			m2prefm1 = n - m1prefm2
			if m1prefm2 == m2prefm1:
				scores[m1] += 0.5
				scores[m2] += 0.5
			elif m1prefm2 > m2prefm1:
				scores[m1] += 1
			else:
				scores[m2] += 1
	winner = np.argwhere(scores == np.max(scores)).flatten().tolist()
	assert tiebreaking == lexicographic_tiebreaking
	a = tiebreaking(None, winner)

	target_winners_scores = list(zip([_ for _ in range(m)], scores))
	target_winners_scores = sorted(target_winners_scores, key = lambda x: x[1], reverse = True)
	assert target_winners_scores[0][0] == a

	sorted_anon_votes = copy.deepcopy(anon_votes)
	sorted_anon_votes = sorted(sorted_anon_votes, key = lambda anon_vote: index_of_winner(anon_vote[1], a))

	mov_k = AppMoVCopeland(a, m, n, n_votes, n_unique, votes, anon_votes)
	print(mov_k, n)
	for k in range(max(1, mov_k), n):
		for b, _ in target_winners_scores:
			if b == a:
				continue
			removed_cnt = np.zeros(len(sorted_anon_votes), dtype = int)
			AI_copeland_witness_is_found = False
			naive_dfs(0, k, sorted_anon_votes, removed_cnt, b, a, tiebreaking)

			if AI_copeland_witness_is_found is True:
				print("GP not satisfies! a = {}, b = {}, k = {}".format(a, b, k))
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
			if "ED-00014" in file:
				continue
			tot += 1
			m, n, n_votes, n_unique, votes, anon_votes = read_preflib_soc("./dataset/" + file)
			assert n == n_votes
			print(file)
			global LP_times, LP_total, LP_cnt, LV_total, LV_cnt
			LP_times = []
			LP_total = 0
			LP_cnt = 0
			LV_total = 0
			LV_cnt = 0
			if AI_copeland(m, n, n_votes, n_unique, votes, anon_votes, lexicographic_tiebreaking) is True:
				cnt += 1
			# print("LP runs {} times, average time is {}.".format(len(LP_times), np.average(LP_times)))
			# print("LP total: {}, cnt: {}".format(LP_total, LP_cnt))
			print("LV total: {}, cnt: {}".format(LV_total, LV_cnt))
	print(cnt, tot, cnt / tot)

if __name__ == "__main__":
	main()
