from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from feat_select import select_features
from feat_select import discretize
import pandas as pd
import csv
import numpy as np
import random
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score


# outputs a classifier and optimal features
def tr_ind(X,y,type,f_sel,seed):
    print('this is the seed: %s'%seed)
    # manually doing grid search cross-validation
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    # kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    fold_feats = []
    for train, test in kf.split(X, y):
        tr_ndx = X.index.values[train]
        X_train, y_train = X.loc[tr_ndx, :], y.loc[tr_ndx, :]

        if type == 'gene' and f_sel == 'chi-squared':
            X_train_copy2 = pd.DataFrame(X_train, copy=True)
            q1 = X_train_copy2.quantile(.25)
            q2 = X_train_copy2.quantile(.5)
            q3 = X_train_copy2.quantile(.75)
            X_train[X_train_copy2 <= q1] = 1
            X_train[(X_train_copy2 > q1) & (X_train_copy2 <= q2)] = 2
            X_train[(X_train_copy2 > q2) & (X_train_copy2 <= q3)] = 3
            X_train[X_train_copy2 > q3] = 4

        # feature selection for train data
        feat_selected = select_features(X_train, y_train, type, f_sel, 50)
        print(feat_selected)
        fold_feats.append(feat_selected)

    # compute average accuracy for each possible parameter combination
    tot_acc = []
    tot_clfs = []
    best_fold = []
    best_params = []

    # parameters to test
    # kernels = ['linear']
    kernels = ['linear','rbf','sigmoid']
    # c_values = [0.1]
    c_values = [.1,1,10,100]
    feat_set_sizes = [5,10,25,50]
    # feat_set_sizes = [20]

    para_list = [(k, c, num_feats) for k in kernels for c in c_values for num_feats in feat_set_sizes]
    for (k, c, num_feats) in para_list:
        count = 0
        acc = []
        clfs = []
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
            # clf = svm.SVC(C=c,gamma="auto",kernel=k,class_weight={0:.3, 1:.7})
            clf = svm.SVC(C=c,gamma="auto",kernel=k,class_weight='balanced')
            # clf = svm.SVC(C=c,gamma="auto",kernel=k)
            # X_train.to_csv('XT.csv')
            # y_train.to_csv('yt.csv')
            # X_test.to_csv('XTe.csv')
            # y_test.to_csv('yte.csv')
            clf.fit(X_train, y_train.values.ravel())
            acc.append(clf.score(X_test,y_test))
            fsc = f1_score(y_test,clf.predict(X_test))
            # print(fsc)
            # acc.append(fsc)
            # pred = clf.decision_function(X_test)
            # c1, c2, _ = roc_curve(y_test.values.ravel(), pred.ravel())
            # area = auc(c1, c2)
            # acc.append(area)
            # print(area)
            clfs.append(clf)
            print(clf.predict(X_test))
            count = count + 1 # iterates to the next fold
        print('kernel:%s, c-value:%f, num feats:%d' % (k, c, num_feats))
        print(np.mean(acc))
        tot_acc.append(np.mean(acc))
        best_params.append([k, c, num_feats])
        tot_clfs.append(clfs)
        best_fold.append(np.argmax(acc))
    mx = np.max(tot_acc)
    print(mx)
    ndx = np.argmax(tot_acc)
    print(best_params[ndx])
    final_pset = best_params[ndx]

    ## selecting best classifier from fold of highest average score
    # clf = tot_clfs[ndx][best_fold[ndx]]
    # feat_selected = fold_feats[best_fold[ndx]][0:final_pset[2]]

    # refitting with best params
    X_copy = pd.DataFrame(X, copy=True)
    y_copy = pd.DataFrame(y, copy=True)
    feat_selected = select_features(X, y, type, f_sel, final_pset[2])
    X_copy = X_copy[feat_selected]  # shrinks to have only selected features
    # clf = svm.SVC(C=final_pset[1], gamma="auto", kernel=final_pset[0], probability=True,class_weight={0:.3, 1:.7})
    clf = svm.SVC(C=final_pset[1], gamma="auto", kernel=final_pset[0], probability=True,class_weight='balanced')
    # clf = svm.SVC(C=final_pset[1], gamma="auto", kernel=final_pset[0], probability=True)
    clf.fit(X_copy, y_copy.values.ravel())
    mx = clf.score(X_copy,y_copy.values.ravel())

    return (clf,feat_selected,mx)











