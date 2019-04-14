import pandas as pd
from feat_select import select_features
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

def tr(X,y,type,f_sel,n_feats,feat_selected):
    fsc = make_scorer(f1_score)
    c_kap = make_scorer(cohen_kappa_score)

    # need to put everything below in a loop so can test different feature set sizes
    X2 = pd.DataFrame(X, copy=True) # copies the original feature dataframe
    y2 = pd.DataFrame(y, copy=True) # copies the original feature dataframe

    # selects top n features
    if feat_selected == 'none':
        feat_selected = select_features(X, y, type, f_sel, n_feats)
        X2 = X2[feat_selected] # shrinks feature matrix to only include selected features
    else:
        X2 = X2[feat_selected]

    # start of classification
    parameters = {'kernel': ('linear','poly','rbf','sigmoid'), 'C': [0.1,.5,1,1.5,2,2.5,3,4,5,6,7,8,9,10,11,12]}
    svc = svm.SVC(gamma="auto",class_weight='balanced')
    # svc = svm.SVC(gamma="auto")
    # clf = GridSearchCV(svc, parameters, cv=4,scoring=fsc)
    clf = GridSearchCV(svc, parameters, cv=4,scoring=c_kap)
    # clf = GridSearchCV(svc, parameters, cv=4)
    clf.fit(X2, y2.values.ravel()) #can also try X here if alter it first

    # for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    #     print(param, score)

    print(clf.best_params_)

    clf = svm.SVC(gamma='auto',class_weight='balanced',C=clf.best_params_['C'],kernel=clf.best_params_['kernel'])
    # clf = svm.SVC(gamma='auto',C=clf.best_params_['C'],kernel=clf.best_params_['kernel'])
    clf.fit(X2, y2.values.ravel())

    return clf, feat_selected