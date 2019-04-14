from train_ind import tr_ind
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def gen_curve(frame,lbl,type,n_times):
    pt_x = []
    pt_y = []
    all_auc = []
    seeds = [8177,3556,1155]
    for i in range(n_times):
        # seed = random.randint(1, 10000)
        seed = seeds[i]
        train_labels, test_labels, train_class, test_class = train_test_split(
            lbl['case_id'], lbl, test_size=0.15, random_state=seed)

        X_train = frame[train_labels].T
        X_test = frame[test_labels].T

        y_train = train_class.set_index('case_id')  # changes first column to be indices
        y_test = test_class.set_index('case_id')  # changes first column to be indices

        y_train[y_train == 1] = 2
        y_train[y_train == 0] = 1
        y_train[y_train == 2] = 0
        y_test[y_test == 1] = 2
        y_test[y_test == 0] = 1
        y_test[y_test == 2] = 0

        if type == 'gene' or type == 'miRNA':
            # normalizing gene expression and miRNA datasets
            X_train_copy = pd.DataFrame(X_train, copy=True)  # copies the original dataframe
            scaler = preprocessing.MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_train = pd.DataFrame(X_train, columns=list(X_train_copy)).set_index(X_train_copy.index.values)

            X_test_copy = pd.DataFrame(X_test, copy=True)  # copies the original dataframe
            X_test = scaler.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=list(X_test_copy)).set_index(X_test_copy.index.values)

        X_train_copy = pd.DataFrame(X_train, copy=True)
        y_train_copy = pd.DataFrame(y_train, copy=True)
        X_test_copy = pd.DataFrame(X_test, copy=True)
        if type == 'CNV':
            clf, feat_selected, mx = tr_ind(X_train_copy, y_train_copy, type, 'chi-squared', seed)
        else:
            clf, feat_selected, mx = tr_ind(X_train_copy, y_train_copy, type, 'ttest', seed)
        X_test_copy = X_test_copy[feat_selected]
        acc = clf.score(X_test_copy,y_test)
        pt_x.append(mx)
        pt_y.append(acc)
        print(clf.decision_function(X_test_copy))
        c1, c2, _ = roc_curve(y_test.values.ravel(), clf.decision_function(X_test_copy).ravel())
        all_auc.append(auc(c1,c2))

    print(all_auc)
    plt.scatter(pt_x,pt_y)
    x = np.linspace(0, 1, 100)
    plt.plot(x,x)
    plt.xlabel('cv accuracy')
    plt.ylabel('validation set accuracy')
    plt.show()
