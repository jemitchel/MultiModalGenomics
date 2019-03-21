from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from feat_select import select_features
import pandas as pd
import csv
import numpy as np
import random

# outputs a classifier and optimal features
def tr_ind(X,y,type):
    # # need to put everything below in a loop so can test different feature set sizes
    # X2 = pd.DataFrame(X, copy=True) # copies the original feature dataframe
    #
    # print(X.shape,y.shape)
    # # selects top n features
    # feat_selected = select_features(X, y, type, 20)
    # X2 = X2[feat_selected] # shrinks feature matrix to only include selected features
    #
    # # start of classification
    # parameters = {'kernel': ('linear','poly','rbf','sigmoid'), 'C': [0.1, 1, 10, 100]}
    # svc = svm.SVC(gamma="auto")
    # clf = GridSearchCV(svc, parameters, cv=4)
    # clf.fit(X2, y.values.ravel()) #can also try X here if alter it first
    #
    # for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    #     print(param, score)



    # manually doing grid search cross-validation
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    c_values = [0.1, 1, 10, 100]
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=12)
    fold_feats = []
    for train, test in kf.split(X, y):
        tr_ndx = X.index.values[train]
        te_ndx = X.index.values[test]
        X_train, X_test = X.loc[tr_ndx, :], X.loc[te_ndx, :]
        y_train, y_test = y.loc[tr_ndx, :], y.loc[te_ndx, :]

        # feature selection for train data
        X_train2 = pd.DataFrame(X_train, copy=True)  # copies the original feature dataframe
        y_train2 = pd.DataFrame(y_train, copy=True)  # copies the original feature dataframe
        feat_selected = select_features(X_train, y_train, type, 30)
        fold_feats.append(feat_selected)
    print(fold_feats)

    # compute average accuracy for each possible parameter combination
    for k in kernels:
        for c in c_values:
            for num_feats in [10,20,30]:
                count = 0
                for train,test in kf.split(X,y):
                    tr_ndx = X.index.values[train]
                    te_ndx = X.index.values[test]
                    X_train, X_test = X.loc[tr_ndx,:], X.loc[te_ndx,:]
                    y_train, y_test = y.loc[tr_ndx,:], y.loc[te_ndx,:]

                    X_train = X_train[fold_feats[count][0:num_feats]] # shrinks train fold to have selected features only
                    X_test = X_test[fold_feats[count][0:num_feats]] # shrinks train fold to have selected features only

                    # start of classification
                    clf = svm.SVC(C=c,gamma="auto",kernel=k)
                    clf.fit(X_train, y_train.values.ravel()) #can also try X_train here if alter it first
                    print('fold number:%d, kernel:%s, c-value:%f, num feats:%d' % (count+1,k,c,num_feats))
                    print(clf.score(X_test,y_test))
                    count = count + 1 # iterates to the next fold








