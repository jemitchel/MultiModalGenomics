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
def tr_ind(X,y,type,f_sel):

    # manually doing grid search cross-validation
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=12)
    fold_feats = []
    for train, test in kf.split(X, y):
        tr_ndx = X.index.values[train]
        X_train, y_train = X.loc[tr_ndx, :], y.loc[tr_ndx, :]

        # feature selection for train data
        if type == 'meth':
            cool = 9
        feat_selected = select_features(X_train, y_train, type, f_sel, 5)
        fold_feats.append(feat_selected)

    # compute average accuracy for each possible parameter combination
    tot_acc = []
    best_params = []

    # parameters to test
    kernels = ['linear','rbf']
    # kernels = ['linear','rbf', 'sigmoid']
    c_values = [0.1,1]
    feat_set_sizes = [2,5]

    para_list = [(k, c, num_feats) for k in kernels for c in c_values for num_feats in feat_set_sizes]
    for (k, c, num_feats) in para_list:
        count = 0
        acc = []
        for train,test in kf.split(X,y):
            tr_ndx = X.index.values[train]
            te_ndx = X.index.values[test]
            X_train, X_test = X.loc[tr_ndx,:], X.loc[te_ndx,:]
            y_train, y_test = y.loc[tr_ndx,:], y.loc[te_ndx,:]

            # X_train, y_train = discretize(X_train,y_train,'miRNA',2.5) #SPECIAL TEST ONLY
            # X_test, y_test = discretize(X_test,y_test,'miRNA',2.5) #SPECIAL TEST ONLY

            X_train = X_train[fold_feats[count][0:num_feats]] # shrinks train fold to have selected features only
            X_test = X_test[fold_feats[count][0:num_feats]] # shrinks train fold to have selected features only

            # start of classification
            clf = svm.SVC(C=c,gamma="auto",kernel=k)
            clf.fit(X_train, y_train.values.ravel())
            acc.append(clf.score(X_test,y_test))
            count = count + 1 # iterates to the next fold
        print('kernel:%s, c-value:%f, num feats:%d' % (k, c, num_feats))
        print(np.mean(acc))
        tot_acc.append(np.mean(acc))
        best_params.append([k, c, num_feats])
    print(np.max(tot_acc))
    ndx = np.argmax(tot_acc)
    # print(ndx)
    print(best_params[ndx])

    final_pset = best_params[ndx]
    feat_selected = select_features(X, y, type, f_sel, final_pset[2])
    X = X[feat_selected]  # shrinks to have only selected features
    clf = svm.SVC(C=final_pset[1], gamma="auto", kernel=final_pset[0], probability=True)
    clf.fit(X, y.values.ravel())

    return (clf,feat_selected)











