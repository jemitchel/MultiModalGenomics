import pandas as pd
from feat_select import select_features
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

def tr(X,y,type,f_sel):
    fsc = make_scorer(f1_score)

    # need to put everything below in a loop so can test different feature set sizes
    X2 = pd.DataFrame(X, copy=True) # copies the original feature dataframe

    # selects top n features
    feat_selected = select_features(X, y, type, f_sel, 15)
    X2 = X2[feat_selected] # shrinks feature matrix to only include selected features

    # start of classification
    parameters = {'kernel': ('linear','poly','rbf','sigmoid'), 'C': [0.1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
    svc = svm.SVC(gamma="auto",class_weight='balanced')
    clf = GridSearchCV(svc, parameters, cv=4,scoring=fsc)
    # clf = GridSearchCV(svc, parameters, cv=4)
    clf.fit(X2, y.values.ravel()) #can also try X here if alter it first

    for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
        print(param, score)

    print(clf.best_params_)

    clf = svm.SVC(gamma='auto',class_weight='balanced',C=clf.best_params_['C'],kernel=clf.best_params_['kernel'])
    clf.fit(X2, y.values.ravel())

    return clf, feat_selected