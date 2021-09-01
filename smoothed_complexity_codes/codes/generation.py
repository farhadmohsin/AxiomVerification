import time
import config
from itertools import permutations
from itertools import combinations
import glob
import random
import math
import os
import linecache
import sys
import signal
import json
import scipy.stats as stats
from scipy.special import comb
import numpy as np


class generation():
    """
    The parent class for all synthetic data generation functions. This class should not be constructed
    directly. All child classes are expected to contain the following variable(s).

    :ivar bool IsStrict: True if generating profiles with strict order.
    :ivar bool IsComplete: True if generating profiles are composed of complete linear orders.
    """
    def __init__(self, IsStrict=True, IsComplete=False, Mode=1):
        self.IsStrict = IsStrict
        self.IsComplete = IsComplete
        self.Mode = Mode


class strict_order(generation):
    """
    The generation class for strict-order profiles.
    For use: e.g. strict_order(4,16,16000,False,2).generation_patch()
    generate 4 soi profiles with 16 candidates and 16000 votes in Mode 2.
    """
    def __init__(self, num, numCands, numVoters, IsComplete, Mode):
        super().__init__(True, IsComplete, Mode)
        self.num = num
        self.numCands = numCands
        self.numVoters = numVoters

    def generation_patch(self, batch_num, start = 0):
        os.chdir(config.data_folder)
        if str(batch_num) not in glob.glob("*"):
            os.mkdir(str(batch_num))
        os.chdir(str(batch_num))

        m = self.numCands
        n = self.numVoters
        extension = 'soc'
        if self.IsComplete is False:
            extension = 'soi'
        folder = 'M'+str(m)+'N'+str(n)+'-soc'
        if folder not in glob.glob("*"):
            os.mkdir(folder)
        os.chdir(folder)
        for i in range(self.num):
            file_name = 'M'+str(m)+'N'+str(n)+'-'+str(start+i+1)+'.'+extension
            f = open(file_name, 'w+')
            txt = self.generation_one_short()
            for j in range(len(txt)):
                f.write(txt[j] + '\n')
            f.close()

    def generation_one(self):
        m = self.numCands
        n = self.numVoters
        cand = []
        for i in range(m):
            cand.append('c'+str(i+1))  # c1, c2, c3, ...
        # print(cand)
        text = [str(m)]
        for i in range(m):
            text.append(str(i+1) + ',' + cand[i])
        # print(text)
        text2 = []
        types = 0
        remaining = n
        while remaining > 0:
            k = m
            if self.IsComplete is False:
                k = random.randint(1, m)
            c = random.sample(list(range(1, m+1)), k)
            counts = 1  # By default counts = 1 (Mode 1)
            if self.Mode == 2:
                ratio = 0.02
                counts = random.randint(1, math.floor(ratio*n))
            elif self.Mode == 3:
                ratio = 0.5
                counts = random.randint(1, max(math.floor(ratio * remaining), 2))
            counts = min(remaining, counts)
            remaining -= counts
            vote = str(counts)+','+str(c).strip('[]').replace(" ", "")
            types += 1
            text2.append(vote)
        check = [str(n) + ',' + str(n) + ',' + str(types)]
        txt = text + check + text2
        # print(txt)
        return txt

    def generation_one_short(self):
        m = self.numCands
        n = self.numVoters
        cand = []
        for i in range(m):
            cand.append('c'+str(i+1))  # c1, c2, c3, ...
        # print(cand)
        text = [str(m)]
        for i in range(m):
            text.append(str(i+1) + ',' + cand[i])
        # print(text)
        text2 = []

        remaining = n
        pref = dict()
        for i in range(n):
            c = tuple(random.sample(list(range(1, m + 1)), m))
            if c in pref.keys():
                pref[c] += 1
            else:
                pref[c] = 1
        types = len(pref.keys())

        for key in pref:
            text2.append(str(pref[key])+','+str(key).strip('())').replace(" ", ""))
        check = [str(n) + ',' + str(n) + ',' + str(types)]
        txt = text + check + text2
        # print(txt)
        return txt
