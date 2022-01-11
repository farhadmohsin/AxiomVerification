import pandas as pd
import numpy as np
from itertools import product, permutations

'''
for mm
    we select a pair of alternatives (a,b)
    a is the winner #a \succ a' is the maximin score
    so #a \succ a' <= #a \succ c for all c
    
    for all b \neq a
    we need the maximin score for b
    assume #b \succ b' is the maximin socre for b
    then #b \succ b' <= b \succ c for all c
    
    finally #a \succ a' >= (or >) b \succ b' for all b
'''

m = 4

alt_adjacent_list = [] # list of other oponent for each oponent, m*(m-1) matrix
for a in range(m):
    alt_adjacent_list.append([i for i in range(m) if i != a])

# the i'th element in product_all_pairs[k] is i's maximin score in combination k
product_alt_pairs = [list(prod) for prod in product(*alt_adjacent_list)]

# an example for a single product_alt_pairs and winner

pap0 = product_alt_pairs[0]

print(list(zip(range(m), pap0)))

for i, j in enumerate(pap0):
    # maximin(i) = #i \succ j
    for k in range(m):
        if k==i or k==j:
            continue
        print(f'{i} \succ {j} <= {i} \succ {k}')

a = 1 # a is the winner
for i, j in enumerate(pap0):
    if i == a:
        continue
    if a > i:
        sign = '>='
    else:
        sign = '>'
    print(f'{a} \succ {pap0[a]} {sign} {i} \succ {j}')
    
# %%

'''
for stv
    m! permutations of eliminitaion
        each lead to a different winner
        e.g. [a,b,c,d] -> a is the winner, d is eliminated first
    we check plurality winner for each stage
    keep track of eliminated alternatives
'''

perms = [list(p) for p in permutations(list(range(m)))]

# example for any perm
p = perms[10]
print(p)
w = p[0] # winner is the first alternative

# we shall have (m-1) rounds

eliminated = []

for rnd in range(m-1):
    # for each set of round, we'll have different ones
    print(f'round {rnd}')
    for i in range(m - rnd - 1):
        if p[i] < p[m - rnd - 1]:
            print(f'score({p[i]}) >= score({p[m - rnd - 1]}) ')
        else:
            print(f'score({p[i]}) > score({p[m - rnd - 1]}) ')
    eliminated.append(p[m - rnd - 1])
    print(f'eliminated: {eliminated}')