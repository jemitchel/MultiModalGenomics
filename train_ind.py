from sklearn import svm
from sklearn.model_selection import GridSearchCV
from feat_select import select_features
import pandas as pd
import csv

# outputs a classifier and optimal features
def tr_ind(X,y,type):
    # need to put everything below in a loop so can test different feature set sizes
    X2 = pd.DataFrame(X, copy=True) # copies the original feature dataframe

    # selects top n features
    feat_selected = select_features(X, y, type, 20)
    X2 = X2[feat_selected] # shrinks feature matrix to only include selected features

    # start of classification
    parameters = {'kernel': ('linear','poly','rbf','sigmoid'), 'C': [0.1, 1, 10, 100]}
    svc = svm.SVC(gamma="auto")
    clf = GridSearchCV(svc, parameters, cv=4)
    clf.fit(X2, y.values.ravel()) #can also try X here if alter it first

    for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
        print(param, score)
#test