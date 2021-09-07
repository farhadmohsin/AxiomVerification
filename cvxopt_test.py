import numpy as np
import cvxopt

c=cvxopt.matrix([0,-1],tc='d')
G=cvxopt.matrix([[-1,1],[3,2],[2,3],[-1,0],[0,-1]],tc='d')
h=cvxopt.matrix([1,12,12,0,0],tc='d')

(status, x) = cvxopt.glpk.ilp(c,G.T,h,I=set([0,1]))
print(status)
print(x[0],x[1]) 
print(sum(c.T*x))
