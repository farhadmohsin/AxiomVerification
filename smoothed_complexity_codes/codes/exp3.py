import prefpy_io
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
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

N = 10000
def read_Xdata(inputfile):
    global N
    dim = 200
    X = zeros([N, dim])
    for i in range(N):
        X[i] = inputfile.readline().strip().split()
    return X

def read_Ydata(inputfile):
    global N
    dim = 10
    Y = zeros([N, dim], dtype=int)
    for i in range(N):
        infomation = inputfile.readline().strip().split(":")
        x = list(infomation[1].split())
        Y[i] = [int(x) for x in x if x]
    return Y

def encodeY(y):
    c = 0
    for i in range(len(y)):
        c += y[i]*(2**i)
    return c


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





    os.chdir('D:\Social Choice\Programming')
    filenames = 'M10_X_trainingdata.txt'
    inf = open(filenames, 'r')
    X_ = read_Xdata(inf)
    # N = 10000
    X= X_[0:N, :]
    # print(size(X,0),size(X,1))
    inf.close()
    filenames = 'M10_y_trainingdata.txt'
    inf = open(filenames, 'r')
    y_ = read_Ydata(inf)
    y = y_[0:N, :]
    # print(size(y, 0), size(y, 1))
    inf.close()
    y_train = []
    for i in range(N):
        y_train.append(encodeY(list(y[i])))
    # print(X[0,:])
    # print(y[0,:])max_iter
    # clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)
    # clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 6), max_iter=1000)#
    # clf = MLPRegressor(activation='tanh', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), max_iter=1000)
    clf = svm.LinearSVC()
    clf.fit(X, y_train)

    # test = 202
    # result = clf.predict(X_[test, :].tolist())
    # # prob = clf.predict_proba(X)
    # # print(prob)
    # y0 = y_[test, :].tolist()
    # print("prediction=%r, y0=%r." %(result, y0))
    # print((list(result[0])==y0))

    error = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for ein in range(N):
        result = list(clf.predict(X_[ein, :].tolist())[0])
        y0 = y_[ein, :].tolist()
        print("prediction=%r, y0=%r." % (result, y0))
        if result != y0:
            error += 1
        for i in range(len(y0)):
            if result[i] == 0 and y0[i] == 0:
                TN += 1
            elif result[i] == 1 and y0[i] == 1:
                TP += 1
            elif result[i] == 1 and y0[i] == 0:
                FP += 1
            elif result[i] == 0 and y0[i] == 1:
                FN += 1
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)
    TNR = TN/(TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("E_in=", error / N, ",precision=", precision, ",recall=", recall, ",TNR=", TNR, ",ACCURACY=", accuracy)

    # error2 = 0
    # for ein in range(N, N+400):
    #     result = list(clf.predict(X_[ein, :].tolist())[0])
    #     y0 = y_[ein, :].tolist()
    #     if result != y0:
    #         error2 += 1
    #
    # print("E_out=", error2 / (400))

    '''
    os.chdir('D:\Social Choice\Programming')
    filenames = 'M10N100_X_trainingdata.txt'
    inf = open(filenames, 'r')
    X_ = read_Xdata(inf)
    N = 500
    X= X_[0:N, :]
    inf.close()
    filenames = 'M10N100_y_trainingdata.txt'
    inf = open(filenames, 'r')
    y_ = read_Ydata(inf)
    y = y_[0:N, :]
    inf.close()
    # print(X[0,:])
    # print(y[0,:])max_iter
    y_train = []
    for i in range(N):
        y_train.append(encodeY(list(y[i])))
    # clf = svm.SVC()
    clf = svm.SVC(degree=3)
    clf.fit(X, y_train)

    # test = 122
    # result = clf.predict(X_[test, :].tolist())[0]
    # # prob = clf.predict_proba(X)
    # # print(prob)
    # y0 = encodeY(y_[test, :].tolist())
    # print("prediction=%r, y0=%r." %(result, y0))
    # print((result==y0))

    error = 0
    for ein in range(N):
        result = clf.predict(X_[ein, :].tolist())[0]
        y0 = encodeY(y_[ein, :].tolist())
        # print("prediction=%r, y0=%r." % (result, y0))
        if result != y0:
            error += 1

    print("E_in=",error/N)

    error2 = 0
    for ein in range(N,900):
        result = clf.predict(X_[ein, :].tolist())[0]
        y0 = encodeY(y_[ein, :].tolist())
        if result != y0:
            error2 += 1

    print("E_out=",error2/(900-N))
    '''
