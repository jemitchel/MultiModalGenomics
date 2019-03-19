import pandas as pd
import pymrmr
import numpy as np

def select_features(X,y,modality):
    # calls helper function to discretize
    X,y = discretize(X,y,modality,1)

    # combine response with features to one dataframe [y,X]
    z = pd.concat([y, X], axis=1)

    # calling mRMR function
    feat_selected = pymrmr.mRMR(z,'MIQ',10)
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
        iqr = (X.quantile(.75) - X.quantile(.25)) * 2
        med = X.median(axis=0)
        X2 = pd.DataFrame(X, copy=True)
        X2[X2 < med - iqr] = np.nan
        X2[X2 > med + iqr] = np.nan
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


