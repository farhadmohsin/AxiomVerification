import numpy as np
from scipy.optimize import linprog
import cvxopt
from time import time

c = [-1, -2]
A = [[2, 1], [-4, 5], [1, -2],]
b = [20, 10, 2]
x0_bounds = (0, None)
x1_bounds = (0, None)

tik = time()
res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
tok = time()

print(res.x, res.message, res.fun, tok-tik)

# c=cvxopt.matrix([0,-1],tc='d')
# G=cvxopt.matrix([[-1,1],[3,2],[2,3],[-1,0],[0,-1]],tc='d')
# h=cvxopt.matrix([1,12,12,0,0],tc='d')

# (status, x) = cvxopt.glpk.ilp(c,G.T,h,I=set([0,1]))
# print(status)
# print(x[0],x[1]) 
# print(sum(c.T*x))

c = cvxopt.matrix(c, tc = 'd')
A = cvxopt.matrix([[2, 1], [-4, 5], [1, -2], [1,1], [-1, 0], [0,-1]], tc = 'd')
b = cvxopt.matrix([20, 10, 2, -1, 0, 0], tc = 'd')

cvxopt.solvers.options['show_progress'] = False
sol = cvxopt.solvers.lp(c, A.T, b)

print(sol['x'])

tik = time()
(status, x) = cvxopt.glpk.ilp(c, A.T, b, I=set(range(len(c))))
tok = time()

print(status)
print(x) 
# print(sum(c.T*x))
print(tok - tik)