import time
import config
from itertools import permutations
from itertools import combinations
import glob
import random
import os
import linecache
import sys
import signal
import json
import scipy.stats as stats
from scipy.special import comb
import numpy as np
from generation import *


if __name__ == '__main__':
    #os.chdir(config.data_folder)
    # for i in range(5, 11):
    #     for j in [2, 3]:
    #         for k in [True, False]:
    #             strict_order(2, i, 10000, k, j).generation_patch()
    # strict_order(100000, 10, 10000, False, 2).generation_patch()
    #txt = strict_order(1, 3, 3, True, 1).generation_one()
    #txt = strict_order(100R, 4, 10, True, 1).generation_patch()
    #strict_order(1000, 10, 1000, True, 1).generation_patch()

    m = int(sys.argv[1])
    nmin = int(sys.argv[2])
    nmax = int(sys.argv[3])
    ind = int(sys.argv[4])
    trials = int(sys.argv[5])
    for n in range(nmin, nmax+1, ind):
        strict_order(trials, m, n, True, 1).generation_patch()
