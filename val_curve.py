from train_ind import tr_ind
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

def gen_curve(X_train,y_train,X_test,y_test,type,n_times):
    pt_x = []
    pt_y = []
    all_auc = []
    for i in range(n_times):
        X_train_copy = pd.DataFrame(X_train, copy=True)
        y_train_copy = pd.DataFrame(y_train, copy=True)
        X_test_copy = pd.DataFrame(X_test, copy=True)
        seed = random.randint(1, 10000)
        clf, feat_selected, mx = tr_ind(X_train_copy,y_train_copy,type,'ttest',seed)
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
