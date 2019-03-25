import pandas as pd
import pymrmr
import numpy as np
from scipy.stats import ttest_ind

def select_features(X,y,modality,n_feats):
    if modality == 'gene':
        X = reduce(X,y)

    # calls helper function to discretize
    X,y = discretize(X,y,modality,2) #4th param is number std away from mean as discretization threshold

    # combine response with features to one dataframe [y,X]
    z = pd.concat([y, X], axis=1)

    # calling mRMR function
    feat_selected = pymrmr.mRMR(z,'MIQ',n_feats)
    return (feat_selected)

def discretize(X,y,modality,n): #features need to be -2,0,2 and response needs to be -1,1
    # discretizes feature data
    if modality == 'CNV':
        X[X == -1] = -2
        X[X == 0] = 0
        X[X == 1] = 2
    else:
        # get trimmed mean and trimmed standard deviation each row column (since features are now columns)
        # gets mean and std of each feature excluding outliers
        q1 = X.quantile(.25)
        q3 = X.quantile(.75)
        add_on = (q3 - q1) * 1.5
        X2 = pd.DataFrame(X, copy=True)
        X2[X2 < q1 - add_on] = np.nan
        X2[X2 > q3 + add_on] = np.nan
        std = X2.std(axis=0)
        av = X2.mean(axis=0)
        # std = X.std(axis=0) # for using non trimmed std
        # av = X.mean(axis=0) # for using non trimmed mean

        X[X < av-(n*std)] = -2
        X[X > av+(n*std)] = 2
        X[abs(X) != 2] = 0
        X = X.astype('int64') #makes the numbers integers, not floats

    # changes discretization for class labels
    y[y == 0] = -1
    y = y.astype('int64')  # makes the numbers integers, not floats
    return (X,y)

def reduce(X,y):
    t1 = X.loc[y['label'] == 0]
    t2 = X.loc[y['label'] == 1]
    stat,pv = ttest_ind(t1,t2,axis=0)
    feats = []
    for i in range(len(pv)):
        if pv[i] < .05:
            feats.append(list(X)[i])
    # pv = [i for i in range(len(pv)) if pv[i] < .1]
    # X = X.iloc[:,pv]
    X = X.loc[:,feats]
    print(X.shape[1])
    return X


