import numpy as np
from voting_utils import *
from bruteforce import *
from itertools import combinations
from scipy.optimize import linprog
import cvxopt
import copy
from time import time
from ortools.sat.python import cp_model

def combs(a, r):
    b = np.fromiter(combinations(a, r), np.dtype([('', a.dtype)]*r))
    return b.view(a.dtype).reshape(-1, r)

def prefab(a, b, ranking):
    """
    Returns
    -------
    Whether a \succ b in ranking
    """
    ind_a = np.argwhere(ranking==a).flatten()[0]
    ind_b = np.argwhere(ranking==b).flatten()[0]
        
    if ind_a < ind_b: # ind_a < ind_b means a is ranked higher
        return 1
    return -1


def Rab(a, b, rankings):
    """
    Parameters
    ----------
    a, b : alternatives
    rankings : list of rankings

    Returns
    -------
    indices of rankings that have a \succ b
    """
    
    ab_indices = [i for i in range(len(rankings)) if (prefab(a,b,rankings[i]) > 0)]    
    return ab_indices

def e(m, i):
    """returns [0 ... 1 ... 0] with e[i] = 1 """
    return [1 if j==i else 0 for j in range(m)]

def neg(a):
    return [-x for x in a]
    
#%%
cvxopt.solvers.options['show_progress'] = False

def create_LP(anon_votes, b, vbw, vwb, C, rankings):
    A = [] # A : need to look into vbw ^^
    h = [] # h: need to look into vwb
    for j in C:
        # print(f'comparing b={b} and j={j}')
        temp = []
        for k in vbw:
            temp.append(-1 * prefab(b, j, rankings[k]))
        A.append(temp)
        
        h0 = 0
        for k in vwb:
            h0 += int(anon_votes[k][0]) * prefab(b, j, rankings[k])
        h.append(h0)
    for l, k in enumerate(vbw):
        A.append(e(len(vbw),l))
        h.append(int(anon_votes[k][0]))
        A.append(neg(e(len(vbw),l)))
        h.append(0)
    
    # print(A)
    # print(h)
    
    A = cvxopt.matrix(A, tc = 'd')
    h = cvxopt.matrix(h, tc = 'd')
    c = cvxopt.matrix([-1 for k in range(len(vbw))], tc = 'd')

    return A, h, c

def create_CP_SAT(anon_votes, b, vbw, vwb, C, rankings):
    A = [] # A : need to look into vbw ^^
    h = [] # h: need to look into vwb
    for j in C:
        # print(f'comparing b={b} and j={j}')
        temp = []
        for k in vbw:
            temp.append(-1 * prefab(b, j, rankings[k]))
        A.append(temp)
        
        h0 = 0
        for k in vwb:
            h0 += int(anon_votes[k][0]) * prefab(b, j, rankings[k])
        h.append(h0)
    
    upper_bounds = []
    for l, k in enumerate(vbw):
        upper_bounds.append(int(anon_votes[k][0]))
            
    return A, h, upper_bounds


def create_new_pref(anon_votes, vbw, absent_x):
    copy_votes = copy.deepcopy(anon_votes)
    for i,ranking in enumerate(vbw):
        copy_votes[vbw[i]][0] = int(absent_x[i])
    return copy_votes

def check_new_pref(anon_votes, votes, b, vbw, x, voting_rule, tiebreaking):
    new_votes = create_full_pref(create_new_pref(anon_votes, vbw, list(x)))
                    
    new_w_, new_s = voting_rule(new_votes)
    new_w = tiebreaking(votes, new_w_)

    # print(f'\t new_w = {new_w}, new_s = {new_s}')
    
    if(new_w == b):
        found_flag = True
        # print(Copeland_winner(votes))
        # print(Copeland_winner(new_votes))
        
        return True, new_votes
    
    return False, []


class VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.solution_list = []

    def on_solution_callback(self):
        self.solution_list.append([self.Value(v) for v in self.__variables])

def ILP_search(votes, anon_votes, n, m, voting_rule, tiebreaking, debug = False):
    
    # initializtion
    w_, s = voting_rule(votes)
    w = tiebreaking(votes, w_)
    
    rankings = [av[1] for av in anon_votes]
    
    # building the LP
    
    # w is the winner
    # say b is the target winner
    
    # getting order of evaluating b, assuming highest scored alternatives are best
    #   as target winner
    s_ranks = np.argsort(-s)
    
    # pre-compute combinations
    alts = np.arange(m-1)
    all_combinations = []
    for i in range((m+1)//2, m):
        all_combinations.append(combs(alts, i))
    
    
    found_flag = False
    for b in s_ranks:
    # for each possible target winner
        # if(found_flag):
        #     break
        if(b == w):
            continue
        vbw = Rab(b, w, rankings)   # ranking indices with b \succ w
        vwb = Rab(w, b, rankings)   # ranking indices with w \succ b
        
        len_x = len(vbw) # no. of variables
        
        for i in range((m+1)//2, m):
            # if(found_flag):
            #     break
            # for different no. head-to-head wins
            i_combs = all_combinations[i-(m+1)//2]
            
            for temp in i_combs:
            # for different combinations of wins
                C = temp.copy()
                C[C==b] = m-1 # replace 
                if(debug):
                    print(f"b = {b}, C = {C}")
                
                # create the LP
                # compute A and h (obj. fun. would be 0 since we just want feasibility) Ax <= h
                # len(x) = len(vbw), because these are the only rankings manipulatable
                A, h, c = create_LP(anon_votes, b, vbw, vwb, C, rankings)
                # if(debug):
                #     if(len(C) == (m+1)//2):
                #         print(A.T)
                #         print(h)
                
                sol = cvxopt.solvers.lp(c, A.T, h)
                # print(sum(c.T * sol['x']))
                if(sol['status'] != 'optimal'):
                    if(debug):
                        print('LP not feasible:', sol['status'])
                    continue
                
                # This is the previously used GLPK ILP codes
                
                if(False):
                
                    (status, x) = cvxopt.glpk.ilp(c, A.T, h, I=set(range(len(c))))
                    # print(sum(c.T*x))
                    if(status != 'optimal'):
                        if(debug):
                            print('ILP not feasible', 'status')
                        continue
                    else:
                        if(debug):
                            print('ILP feasible')
                        pass
                    
                    # probably need to check all feasible cases here
                    
                    new_votes = create_full_pref(create_new_pref(anon_votes, vbw, list(x)))
                    
                    new_w_, new_s = voting_rule(new_votes)
                    new_w = tiebreaking(votes, new_w_)
                    
                    print(majority_graph(new_votes))
                
                    # print(f'\t new_w = {new_w}, new_s = {new_s}')
                    
                    if(new_w == b):
                        found_flag = True
                        print("GP not saitsified")
                        print(Copeland_winner(votes))
                        print(Copeland_winner(new_votes))
                        if(debug):
                            return found_flag, new_votes
                        else:
                            return found_flag
                        
                ## ortools ILP codes start
                
                A, h, upper_bounds = create_CP_SAT(anon_votes, b, vbw, vwb, C, rankings)
                if(debug):
                    print(f'A = {A}')
                    print(f'h = {h}')
                    print(f'ub = {upper_bounds}')

                
                num_vals = len(vbw)
                x = []
                
                model = cp_model.CpModel()
                for i in range(num_vals):
                    x.append(model.NewIntVar(0, upper_bounds[i], f'x[{i}]'))
            
                # Create the constraints.
                
                for i in range(len(h)):
                    constraint_expr = [A[i][k] * x[k] for k in range(num_vals)]
                    model.Add(sum(constraint_expr) <= h[i])
                
                # Create a solver and solve.
                solver = cp_model.CpSolver()
                solution_collector = VarArraySolutionCollector(x)
                status = solver.SearchForAllSolutions(model, solution_collector)
            
                if(debug):
                    print('Status = %s' % solver.StatusName(status))
                    # print('All solutions:', solution_collector.solution_list)
                    print('Number of solutions found: %i' % len(solution_collector.solution_list))
                
                for x in solution_collector.solution_list:
                    flag, new_votes = check_new_pref(anon_votes, votes, b, vbw, x, voting_rule, tiebreaking)
                    if(flag):
                        found_flag = True
                        print(f"GP not satsified, b = {b}")
                            
                        if(debug):
                            print("original:", Copeland_winner(votes))
                            print("new:", Copeland_winner(new_votes))
                            return found_flag, new_votes
                        else:
                            return found_flag
                
                     
    # print(f"reached end, flag is {found_flag}")
    return found_flag

#%%

if __name__ == '__main__':
        
    n = 30
    m = 4
    
    voting_rule = Copeland_winner
    tiebreaking = lexicographic_tiebreaking
    
    np.random.seed(0)
    
    brute_cnt = 0
    ILP_cnt = 0
    
    cnt = 0 
    
    brute_times = []
    ILP_times = []
    
    samples = 100
    for s in range(samples):
        print(cnt)
        cnt += 1
        votes = gen_pref_profile(n, m)
        m, n, n_votes, n_unique, anon_votes = anonymize_pref_profile(votes)
        # anon_votes = np.array(anon_votes)
        
        # tik = time()
        # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, voting_rule, tiebreaking)):
        #     brute_cnt += 1
        # tok = time()
        # brute_times.append(tok-tik)
        
        tik = time()
        if(ILP_search(votes, anon_votes, n, m, voting_rule, tiebreaking)):
            ILP_cnt += 1
            pp = copy.deepcopy(votes)
            anon_pp = copy.deepcopy(anon_votes)
        tok = time()
        ILP_times.append(tok-tik)    
        
        # if(brute_force(m, n, n_votes, n_unique, votes, anon_votes, voting_rule, tiebreaking)):
        #     if(not(ILP_search(votes, anon_votes, n, m, voting_rule, tiebreaking))):
                # break
        
    print(brute_cnt, ILP_cnt)
    print(np.mean(brute_times), np.mean(ILP_times))
