# import pymrmr
# import mifs
import mrmr

def select_features(X,y):
    ranks = mrmr(X,y,6)
    # # define MI_FS feature selection method
    # feat_selector = mifs.MutualInformationFeatureSelector('MRMR')
    #
    # # find all relevant features
    # feat_selector.fit(X, y)
    #
    # # check selected features
    # selected = feat_selector.support_
    #
    # # check ranking of features
    # ranks = feat_selector.ranking_
    #
    # # # call transform() on X to filter it down to selected features
    # # X_filtered = feat_selector.transform(X)

    return (ranks)