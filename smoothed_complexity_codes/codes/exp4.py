import prefpy_io
import structures_py3
import math
import os
import itertools
# from .preference import Preference
from numpy import *
from profile import Profile
from mechanism import *
import glob
import mov
# from mov import *
from preference import Preference
import matplotlib.pyplot as plt
from sklearn.neural_network import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

import warnings
warnings.filterwarnings("ignore")

def read_Xdata(inputfile):
    N = 4000
    dim = 200
    X = zeros([N, dim])
    for i in range(N):
        X[i] = inputfile.readline().strip().split()
    return X

def read_Ydata(inputfile):
    N = 10000
    dim = 10
    Y = zeros([N, dim], dtype=int)
    for i in range(N):
        Y[i] = inputfile.readline().strip().split()
    return Y
def encodeY(y):
    c = 0
    for i in range(len(y)):
        c += y[i]*(2**i)
    return c

def comp_soc_folder(pname, foldername, m):
    STV_vector = []
    RP_vector = []
    for i in range(m):
            STV_vector.append(0)
            RP_vector.append(0)
    filenames = glob.glob(pname+"/"+foldername+"/*")
    for inputfile in filenames:
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)

        rec = inputfile.strip()

        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        # veto = MechanismVeto().getWinners(profile)
        # print("veto=", veto)
        # pwr = MechanismPluralityRunOff().PluRunOff_cowinners(profile)
        # print("pwr=", pwr)
        # pwr_mov = MechanismPluralityRunOff().getMov(profile)
        # Borda_mov = MechanismBorda().getMov(profile)
        STVwinners = MechanismSTV().STVwinners(profile)
        # print(inputfile, "STV winners=", STVwinners)
        RPwinners = MechanismRankedPairs().getWinners(profile)
        STV_vector[len(STVwinners)-1] += 1
        RP_vector[len(SRPwinners) - 1] += 1
    return STV_vector, RP_vector

if __name__ == '__main__':
    # X = [[0., 0.], [1., 1.]]
    # y = [0, 1]
    # clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(X, y)
    # result = clf.predict([[2., 2.], [-1., -2.]])
    # print("prediction=",result)


    # X = [[0, 0], [1, 1]]
    # y = [0, 1]
    # clf = svm.SVC()
    # clf.fit(X, y)
    # result = clf.predict([[2., 2.]])
    # print("prediction=", result)





    # os.chdir('D:\Social Choice\Programming')
    # filenames = 'M10_X_trainingdata.txt'
    # inf = open(filenames, 'r')
    # X_ = read_Xdata(inf)
    # N = 10000
    # X= X_[0:N, :]
    # inf.close()
    # filenames = 'M10_y_trainingdata.txt'
    # inf = open(filenames, 'r')
    # y_ = read_Ydata(inf)
    # countone = 0
    # for i in range(N):
    #     for j in range(10):
    #         if y_[i][j]==1:
    #             countone += 1
    # print(countone/N)
    os.chdir(structures_py3.path)
    filenames = glob.glob("M5N5-*.csv")
    # filenames = filenames[16745:16745+100]
    for inputfile in filenames:
        inf = open(inputfile, 'r')
        cmap, rmaps, rmapscounts, nvoters = prefpy_io.read_election_file(inf)

        rec = inputfile.strip()

        inf.close()
        profile = Profile(cmap, preferences=[])
        Profile.importPreflibFile(profile, inputfile)

        # veto = MechanismVeto().getWinners(profile)
        # print("veto=", veto)
        # pwr = MechanismPluralityRunOff().PluRunOff_cowinners(profile)
        # print("pwr=", pwr)
        # pwr_mov = MechanismPluralityRunOff().getMov(profile)
        # Borda_mov = MechanismBorda().getMov(profile)
        STVwinners = MechanismSTV().STVwinners(profile)
        # print(inputfile, "STV winners=", STVwinners)
        RPwinners = MechanismRankedPairs().getWinners(profile)
        print(inputfile, "STV winners=", STVwinners, "RP winners=", RPwinners)




