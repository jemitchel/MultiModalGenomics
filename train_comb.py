from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from feat_select import select_features
from feat_select import discretize
import pandas as pd
import csv
import numpy as np
import random

# outputs a classifier and optimal features
def tr_comb(X,y):

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=12)
    # parameters to test
    kernels = ['linear', 'rbf', 'sigmoid']
    c_values = [0.1, 1, 10]

    para_list = [(k, c) for k in kernels for c in c_values]
    tot_acc = []
    best_params = []
    for (k, c) in para_list:
        acc = []
        for train, test in kf.split(X, y):
            tr_ndx = X.index.values[train]
            te_ndx = X.index.values[test]
            X_train, X_test = X.loc[tr_ndx, :], X.loc[te_ndx, :]
            y_train, y_test = y.loc[tr_ndx, :], y.loc[te_ndx, :]

            # start of classification
            clf = svm.SVC(C=c, gamma="auto", kernel=k)
            clf.fit(X_train, y_train.values.ravel())
            acc.append(clf.score(X_test,y_test))
        tot_acc.append(np.mean(acc))
        best_params.append([k, c])

    ndx = np.argmax(tot_acc)
    final_pset = best_params[ndx]
    clf = svm.SVC(C=final_pset[1], gamma="auto", kernel=final_pset[0], probability=True)
    clf.fit(X, y.values.ravel())

    return (clf)

