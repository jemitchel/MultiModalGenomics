import pandas as pd
import os
from sklearn.model_selection import train_test_split
from train_ind import tr_ind
from sklearn import preprocessing
from feat_select import select_features
from train_comb import tr_comb
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import numpy as np
from sklearn import svm
from val_curve import gen_curve

def pipeline(rem_zeros):

    # loads all data
    os.chdir("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data")
    labels = pd.read_csv('Label_selected.csv')
    ndx = labels.index[labels['label'] > -1].tolist() #gets indices of patients to use
    lbl = labels.iloc[ndx,[0,4]] #makes a new dataframe of patients to use (their IDs and survival response)

    os.chdir("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\Processed_data")
    gene = pd.read_csv('gene.csv',index_col=0)
    miRNA = pd.read_csv('miRNA.csv',index_col=0)
    meth = pd.read_csv('meth.csv',index_col=0)
    CNV = pd.read_csv('CNV.csv',index_col=0)


    # optionally removes rows (features) that are all 0 across patients
    if rem_zeros == True:
        gene = gene.loc[~(gene==0).all(axis=1)]
        miRNA = miRNA.loc[~(miRNA==0).all(axis=1)]

    # splitting labels into train set and validation set
    train_labels, test_labels, train_class, test_class = train_test_split(
        lbl['case_id'], lbl, test_size=0.15, random_state=42)

    # divides individual modalities into same train and test sets and transposes data frames
    miRNA_train = miRNA[train_labels].T
    miRNA_test = miRNA[test_labels].T
    gene_train = gene[train_labels].T
    gene_test = gene[test_labels].T
    CNV_train = CNV[train_labels].T
    CNV_test = CNV[test_labels].T
    meth_train = meth[train_labels].T
    meth_test = meth[test_labels].T




    # normalizing gene expression and miRNA datasets
    miRNA_train_copy = pd.DataFrame(miRNA_train, copy=True) # copies the original dataframe
    miRNA_scaler = preprocessing.MinMaxScaler().fit(miRNA_train)
    miRNA_train = miRNA_scaler.transform(miRNA_train)
    miRNA_train = pd.DataFrame(miRNA_train,columns=list(miRNA_train_copy)).set_index(miRNA_train_copy.index.values)

    gene_train_copy = pd.DataFrame(gene_train, copy=True) # copies the original dataframe
    # gene_scaler = preprocessing.QuantileTransformer(output_distribution='normal').fit(gene_train)
    gene_scaler = preprocessing.MinMaxScaler().fit(gene_train)
    gene_train = gene_scaler.transform(gene_train)
    gene_train = pd.DataFrame(gene_train,columns=list(gene_train_copy)).set_index(gene_train_copy.index.values)

    gene_test_copy = pd.DataFrame(gene_test, copy=True)  # copies the original dataframe
    gene_test = gene_scaler.transform(gene_test)
    gene_test = pd.DataFrame(gene_test, columns=list(gene_test_copy)).set_index(gene_test_copy.index.values)

    train_class = train_class.set_index('case_id') # changes first column to be indices
    test_class = test_class.set_index('case_id') # changes first column to be indices

    # for doing quantile scaling instead
    q1 = gene_train.quantile(.25)
    q2 = gene_train.quantile(.5)
    q3 = gene_train.quantile(.75)
    gene_train_copy2 = pd.DataFrame(gene_train, copy=True)
    gene_train_copy2[gene_train <= q1] = -2
    gene_train_copy2[(gene_train > q1) & (gene_train <= q2)] = -1
    gene_train_copy2[(gene_train > q2) & (gene_train <= q3)] = 1
    gene_train_copy2[gene_train > q3] = 2


    # # does this to over-represent the minority class
    # train_class_copy = pd.DataFrame(train_class, copy=True)
    # sm = SMOTE(random_state=42)
    # gene_train, train_class = sm.fit_resample(gene_train, train_class.values.ravel())
    # gene_train = pd.DataFrame(gene_train,columns=list(gene_train_copy))
    # train_class = pd.DataFrame(train_class,columns=['label'])


    # makes copies of the y dataframe because tr_ind alters it
    train_class_copy1,train_class_copy2,train_class_copy3,train_class_copy4,train_class_copy5 = pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True)



    gene_train_copy3 = pd.DataFrame(gene_train_copy2, copy=True)
    # miRNA_train_copy2 = pd.DataFrame(miRNA_train, copy=True)
    # miRNA_train_copy2.to_csv('test.csv')
    # meth_train_copy2 = pd.DataFrame(meth_train, copy=True)
    # CNV_train_copy2 = pd.DataFrame(CNV_train, copy=True)

    # gen_curve(gene_train,train_class,gene_test,test_class,'gene',5)
    # # do cross validation to get best classifiers and feature sets for each modality
    # clf_gene, fea_gene = tr_ind(gene_train,train_class_copy1,'gene','ttest')
    # clf_miRNA, fea_miRNA = tr_ind(miRNA_train,train_class_copy2,'miRNA','mrmr')
    # clf_meth, fea_meth = tr_ind(meth_train,train_class_copy3,'meth','mrmr')
    # clf_CNV, fea_CNV = tr_ind(CNV_train,train_class_copy4,'CNV','chi-squared')

    feat_selected = select_features(gene_train_copy2, train_class_copy1, 'gene', 'chi-squared', 10)
    gene_train_copy3 = gene_train_copy3[feat_selected]
    # clf = svm.SVC(C=100, gamma="auto", kernel='rbf')
    # clf.fit(gene_train_copy2, train_class_copy5.values.ravel())
    # gene_test = gene_test[feat_selected]
    # print(clf.decision_function(gene_test))
    # print(clf.predict(gene_test))
    # print(clf.score(gene_test,test_class))
    # c1,c2,_ = roc_curve(test_class.values.ravel(), clf.decision_function(gene_test).ravel())
    # print(auc(c1, c2))

    os.chdir("D:\\4813")
    gene_train_copy3.to_csv('feature_vis4.csv')
    train_class.to_csv('group4.csv')
    #
    # # select features
    # miRNA_train_copy2 = miRNA_train_copy2[fea_miRNA]
    # gene_train_copy2 = gene_train_copy2[fea_gene]
    # meth_train_copy2 = meth_train_copy2[fea_meth]
    # CNV_train_copy2 = CNV_train_copy2[fea_CNV]
    # gene_test = gene_test[fea_gene]
    #
    # pred_miRNA = clf_miRNA.decision_function(miRNA_train_copy2)
    # pred_gene = clf_gene.decision_function(gene_train_copy2)
    # pred_meth = clf_meth.decision_function(meth_train_copy2)
    # pred_CNV = clf_CNV.decision_function(CNV_train_copy2)
    # pred_gene = clf_gene.decision_function(gene_test)
    # print(pred_gene)
    # print(clf_gene.predict(gene_test))
    # c1,c2,_ = roc_curve(test_class.values.ravel(), pred_gene.ravel())
    # print(auc(c1, c2))
    # print(clf_gene.score(gene_test,test_class))
    #
    # new_feats = {'sample':miRNA_train.index.values,'miRNA':pred_miRNA, 'gene':pred_gene, 'meth':pred_meth, 'CNV':pred_CNV}
    # new_feats = pd.DataFrame(data=new_feats)
    # new_feats = new_feats.set_index('sample')
    # print(new_feats)
    # new_feats.to_csv('new_feats.csv')

    # tr_comb(new_feats,train_class_copy5)


    # fts = pd.read_csv('new_feats.csv')
    # c1,c2,_ = roc_curve(train_class_copy5.values.ravel(), fts['gene'].ravel())
    # print(auc(c1, c2))

    # # for plotting histograms
    # feat = select_features(gene_train,train_class_copy5,'gene','ttest',10)
    # t1 = gene_train_copy2.loc[train_class['label'] == 0]
    # t2 = gene_train_copy2.loc[train_class['label'] == 1]
    # # t1[feat].to_csv('t1.csv')
    # # t2[feat].to_csv('t2.csv')
    # t1 = t1[feat[1]]
    # t2 = t2[feat[1]]
    # plt.hist([t1,t2],normed=True)
    # stat, pv = ttest_ind(t1, t2)
    # # stat,pv = mannwhitneyu(t1,t2)
    # print(pv)
    # # plt.hist(t2,add=True)
    # # t1.to_csv('t1.csv')
    # # t2.to_csv('t2.csv')
    # plt.show()


    # pvals = []
    # t1 = gene_train_copy2.loc[train_class['label'] == 0]
    # t2 = gene_train_copy2.loc[train_class['label'] == 1]
    # for i in range(gene_train.shape[1]):
    #     if i % 10 == 0:
    #         print(i)
    #     if t1.values==t2.values:
    #         print(i)
    #         break
    #     stat,pv = mannwhitneyu(t1.iloc[:,i],t2.iloc[:,i])
    #     pvals.append(pv)
    # # n_feats = 50
    # # feats = []
    # # indicies = np.argsort(pvals)
    # # for i in range(len(indicies)):
    # #     if i < n_feats:
    # #         feats.append(list(X)[indicies[i]])
    # # return feats
    # print(pvals)
    # # print(t1.iloc[:,70:80])
    # # print(t2.iloc[:,70:80])


pipeline(True)






