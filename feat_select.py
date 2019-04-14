import pandas as pd
import pymrmr
import numpy as np
from scipy.stats import ttest_ind
# from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold

def select_features(X,y,modality,method,n_feats):

    if method == 'mrmr':
        if modality == 'gene' or modality == 'meth': #for these doing prefiltering with ttest
            # selector = VarianceThreshold(threshold=.025)
            # selector.fit(X)
            # ndx = selector.get_support(indices=True)
            # feat_keep = []
            # for i in range(X.shape[1]):
            #     if i in ndx:
            #         feat_keep.append(list(X)[i])
            # X = X.loc[:, feat_keep]

            init_feats = reduce(X, y, 2000)
            X = X.loc[:, init_feats]
        elif modality == 'CNV':
            init_feats = chi(X, y, 2000)
            X = X.loc[:, init_feats]

        # calls helper function to discretize
        X,y = discretize(X,y,modality,.5) #4th param is number std away from mean as discretization threshold

        # combine response with features to one dataframe [y,X]
        z = pd.concat([y, X], axis=1)

        # calling mRMR function
        feat_selected = pymrmr.mRMR(z,'MIQ',n_feats)
    elif method == 'ttest':
        feat_selected = reduce(X, y, n_feats)
    elif method == 'chi-squared':
        X,y = discretize(X,y,modality,1)
        feat_selected = chi(X, y, n_feats)
    elif method == 'minfo':
        selector = VarianceThreshold(threshold=.03)
        selector.fit(X)
        ndx = selector.get_support(indices=True)
        feat_keep = []
        for i in range(X.shape[1]):
            if i in ndx:
                feat_keep.append(list(X)[i])
        X = X.loc[:,feat_keep]

        if modality != 'CNV':
            X, y = discretize(X, y, modality, .5)

        feat_selected = minfo(X,y,n_feats)

    return (feat_selected)

def discretize(X,y,modality,n): #features need to be -2,0,2 and response needs to be -1,1
    # discretizes feature data
    if modality == 'CNV':
        X2 = pd.DataFrame(X, copy=True)
        X[X2 == -1] = 0
        X[X2 == 0] = 1
        X[X2 == 1] = 2
    else:
        # get trimmed mean and trimmed standard deviation each row column (since features are now columns)
        # gets mean and std of each feature excluding outliers
        # q1 = X.quantile(.25)
        # q3 = X.quantile(.75)
        # add_on = (q3 - q1) * 1.5
        # X2 = pd.DataFrame(X, copy=True)
        # X2[X2 < q1 - add_on] = np.nan
        # X2[X2 > q3 + add_on] = np.nan
        # std = X2.std(axis=0)
        # av = X2.mean(axis=0)
        std = X.std(axis=0) # for using non trimmed std
        av = X.mean(axis=0) # for using non trimmed mean

        X[X < av-(n*std)] = -2
        X[X > av+(n*std)] = 2
        X[abs(X) != 2] = 0
        X = X.astype('int64') #makes the numbers integers, not floats

    # changes discretization for class labels
    y[y == 0] = -1
    y = y.astype('int64')  # makes the numbers integers, not floats
    return (X,y)

def reduce(X,y,n_feats):
    t1 = X.loc[y['label'] == 0]
    t2 = X.loc[y['label'] == 1]
    stat,pv = ttest_ind(t1,t2,axis=0)
    feats = []
    indicies = np.argsort(pv)
    for i in range(len(indicies)):
        if i < n_feats:
            feats.append(list(X)[indicies[i]])
    return feats

def chi(X,y,n_feats):
    # pvals = []
    # #create contingency tables and run chi sq test
    # for i in range(X.shape[1]):
    #     if i%10 == 0:
    #         print(i)
    #     c_table = pd.crosstab(y['label'],X.iloc[:,i])
    #     pv = chi2_contingency(c_table)[1]
    #     pvals.append(pv)
    #
    # feats = []
    # indicies = np.argsort(pvals)
    # for i in range(len(indicies)):
    #     if i < n_feats:
    #         feats.append(list(X)[indicies[i]])

    _,pvals = chi2(X,y)
    feats = []
    indicies = np.argsort(pvals)
    for i in range(len(indicies)):
        if i < n_feats:
            feats.append(list(X)[indicies[i]])
            # print(pvals[indicies[i]])
    return feats



    # stat,pv = ttest_ind(t1,t2,axis=0)
    # feats = []
    # indicies = np.argsort(pv)
    # for i in range(len(indicies)):
    #     if i < n_feats:
    #         feats.append(list(X)[indicies[i]])
    # return feats

def minfo(X,y,n_feats):
    score = mutual_info_classif(X,y.values.ravel())
    feats = []
    indicies = np.argsort(score)
    indicies = indicies[::-1]
    for i in range(len(indicies)):
        if i < n_feats:
            feats.append(list(X)[indicies[i]])
        else:
            break
    return feats

