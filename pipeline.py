import pandas as pd
import os
from sklearn.model_selection import train_test_split
from train_ind import tr_ind
from sklearn import preprocessing
from feat_select import select_features
from train_comb import tr_comb
# from train_comb import maj_vote
# from train_comb import weight_vote
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import numpy as np
from sklearn import svm
# from val_curve import gen_curve
from val_curve2 import gen_curve
from cv_2 import tr
from feat_curve import make_feat_curve
from cv_2_w_feats import do_cv


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
        lbl['case_id'], lbl, test_size=0.2, random_state=42)

    # removes features (rows) that have any na in them
    meth = meth.dropna(axis='rows')
    miRNA = miRNA.dropna(axis='rows')
    gene = gene.dropna(axis='rows')
    CNV = CNV.dropna(axis='rows')


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

    miRNA_test_copy = pd.DataFrame(miRNA_test, copy=True)  # copies the original dataframe
    miRNA_test = miRNA_scaler.transform(miRNA_test)
    miRNA_test = pd.DataFrame(miRNA_test, columns=list(miRNA_test_copy)).set_index(miRNA_test_copy.index.values)

    gene_train_copy = pd.DataFrame(gene_train, copy=True) # copies the original dataframe
    gene_scaler = preprocessing.MinMaxScaler().fit(gene_train)
    gene_train = gene_scaler.transform(gene_train)
    gene_train = pd.DataFrame(gene_train,columns=list(gene_train_copy)).set_index(gene_train_copy.index.values)
    #
    gene_test_copy = pd.DataFrame(gene_test, copy=True)  # copies the original dataframe
    gene_test = gene_scaler.transform(gene_test)
    gene_test = pd.DataFrame(gene_test, columns=list(gene_test_copy)).set_index(gene_test_copy.index.values)

    # meth_train_copy = pd.DataFrame(meth_train, copy=True) # copies the original dataframe
    # meth_scaler = preprocessing.MinMaxScaler().fit(meth_train)
    # meth_train = meth_scaler.transform(meth_train)
    # meth_train = pd.DataFrame(meth_train,columns=list(meth_train_copy)).set_index(meth_train_copy.index.values)
    #
    # meth_test_copy = pd.DataFrame(meth_test, copy=True)  # copies the original dataframe
    # meth_test = meth_scaler.transform(meth_test)
    # meth_test = pd.DataFrame(meth_test, columns=list(meth_test_copy)).set_index(meth_test_copy.index.values)

    train_class = train_class.set_index('case_id') # changes first column to be indices
    test_class = test_class.set_index('case_id') # changes first column to be indices


    train_class[train_class == 1] = 2
    train_class[train_class == 0] = 1
    train_class[train_class == 2] = 0
    test_class[test_class == 1] = 2
    test_class[test_class == 0] = 1
    test_class[test_class == 2] = 0


    # # for doing quantile scaling instead
    # q1 = gene_train.quantile(.25)
    # q2 = gene_train.quantile(.5)
    # q3 = gene_train.quantile(.75)
    # gene_train_copy2 = pd.DataFrame(gene_train, copy=True)
    # gene_train_copy2[gene_train <= q1] = 1
    # gene_train_copy2[(gene_train > q1) & (gene_train <= q2)] = 2
    # gene_train_copy2[(gene_train > q2) & (gene_train <= q3)] = 3
    # gene_train_copy2[gene_train > q3] = 4



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




    # gen_curve(meth,lbl,'miRNA',3)
    miRNA_train_copy2 = pd.DataFrame(miRNA_train, copy=True)
    meth_train_copy2 = pd.DataFrame(meth_train, copy=True)
    CNV_train_copy2 = pd.DataFrame(CNV_train, copy=True)
    gene_train_copy2 = pd.DataFrame(gene_train, copy=True)

    #
    # # gen_curve(gene_train,train_class,gene_test,test_class,'gene',3)
    # # gen_curve(miRNA_train,train_class,miRNA_test,test_class,'miRNA',4)
    # # gen_curve(CNV_train,train_class,CNV_test,test_class,'CNV',4)
    # # gen_curve(meth_train,train_class,meth_test,test_class,'meth',4)
    # # # do cross validation to get best classifiers and feature sets for each modality

    # # make_feat_curve(gene_train,train_class,gene_test,test_class,'mrmr','gene')
    clf_gene, fea_gene,_ = do_cv(gene_train,train_class_copy1,gene_test,test_class,'ttest','gene',200)
    clf_miRNA, fea_miRNA,_ = do_cv(miRNA_train,train_class_copy2,miRNA_test,test_class,'minfo','miRNA',120)
    clf_meth, fea_meth,_ = do_cv(meth_train,train_class_copy3,meth_test,test_class,'minfo','meth',120)
    clf_CNV, fea_CNV,_ = do_cv(CNV_train,train_class_copy4,CNV_test,test_class,'minfo','CNV',120)


    # clf_gene, fea_gene, mx = tr_ind(gene_train,train_class_copy1,'gene','ttest',5)
    # clf_gene, fea_gene, mx = tr(gene_train,train_class_copy1,'gene','ttest')
    # clf_gene, fea_gene = tr(gene_train,train_class_copy1,'gene','ttest',20,'none')

    # clf_miRNA, fea_miRNA, _ = tr_ind(miRNA_train,train_class_copy2,'miRNA','ttest',5)
    # clf_meth, fea_meth, _ = tr_ind(meth_train,train_class_copy3,'meth','ttest',5)
    # clf_CNV, fea_CNV, _ = tr_ind(CNV_train,train_class_copy4,'CNV','chi-squared',5)
    # feat = select_features(gene_train, train_class, 'gene', 'ttest', 20)
    # print(feat)
    # # print(feat)
    # # print(feat)
    # # feats = select_features(CNV_train, train_class, 'CNV', 'chi-squared', 10)
    # # print(feats)
    # # gene_train_copy3 = gene_train_copy3[feat_selected]
    # # clf = svm.SVC(C=100, gamma="auto", kernel='rbf')
    # # clf.fit(gene_train_copy2, train_class_copy5.values.ravel())

    # #stuff for test
    # gene_train_copy2 = gene_train_copy2[fea_gene]
    # gene_test = gene_test[fea_gene]
    # print(clf_gene.decision_function(gene_test))
    # print(test_class)
    # # print(clf_gene.decision_function(gene_train_copy2))
    # # print(train_class_copy2)
    # # print(clf_gene.predict(gene_test))
    # print('test accuracy:',clf_gene.score(gene_test,test_class))
    # print('train accuracy:',clf_gene.score(gene_train_copy2,train_class_copy2))
    # # print('train accuracy:',mx)
    # c1,c2,_ = roc_curve(test_class.values.ravel(), clf_gene.decision_function(gene_test).ravel())
    # print('auc:',auc(c1, c2))
    # c1, c2, _ = roc_curve(train_class.values.ravel(), clf_gene.decision_function(gene_train_copy2).ravel())
    # print('auc:', auc(c1, c2))
    #
    # # os.chdir("D:\\4813")
    # # gene_train_copy3.to_csv('feature_vis4.csv')
    # # train_class.to_csv('group4.csv')
    # #
    # select features
    # miRNA_train_copy2 = miRNA_train_copy2[fea_miRNA]
    # gene_train_copy2 = gene_train_copy2[fea_gene]
    # meth_train_copy2 = meth_train_copy2[fea_meth]
    # CNV_train_copy2 = CNV_train_copy2[fea_CNV]
    # # gene_test = gene_test[fea_gene]
    # #
    # pred_miRNA = clf_miRNA.decision_function(miRNA_train_copy2)
    # pred_gene = clf_gene.decision_function(gene_train_copy2)
    # pred_meth = clf_meth.decision_function(meth_train_copy2)
    # pred_CNV = clf_CNV.decision_function(CNV_train_copy2)

    # produces training data from part of test data
    miRNA_val1 = miRNA_test.iloc[0:20,:][fea_miRNA]
    gene_val1 = gene_test.iloc[0:20,:][fea_gene]
    meth_val1 = meth_test.iloc[0:20,:][fea_meth]
    CNV_val1 = CNV_test.iloc[0:20,:][fea_CNV]

    val1_class = test_class.iloc[0:20,:]

    miRNA_val2 = miRNA_test.iloc[20:, :][fea_miRNA]
    gene_val2 = gene_test.iloc[20:, :][fea_gene]
    meth_val2 = meth_test.iloc[20:, :][fea_meth]
    CNV_val2 = CNV_test.iloc[20:, :][fea_CNV]

    val2_class = test_class.iloc[20:,:]


    pred_miRNA = clf_miRNA.decision_function(miRNA_val1)
    pred_gene = clf_gene.decision_function(gene_val1)
    pred_meth = clf_meth.decision_function(meth_val1)
    pred_CNV = clf_CNV.decision_function(CNV_val1)

    new_feats = {'sample': miRNA_val1.index.values, 'miRNA': miRNA_val1, 'gene': gene_val1, 'meth': meth_val1,
                 'CNV': CNV_val1}
    # new_feats = {'sample':miRNA_train.index.values,'gene':pred_gene, 'CNV':pred_CNV, 'meth':pred_meth}
    new_feats = pd.DataFrame(data=new_feats)
    new_feats = new_feats.set_index('sample')
    print(new_feats)
    # print(new_feats.head())

    clf = tr_comb(new_feats,val1_class)

    pred_miRNA = clf_miRNA.decision_function(miRNA_val2)
    pred_gene = clf_gene.decision_function(gene_val2)
    pred_meth = clf_meth.decision_function(meth_val2)
    pred_CNV = clf_CNV.decision_function(CNV_val2)

    new_feats_val = {'sample': miRNA_val1.index.values, 'miRNA': miRNA_val1, 'gene': gene_val1, 'meth': meth_val1,
                 'CNV': CNV_val1}

    clf.score(new_feats_val,val2_class)

    # new_feats = {'sample':miRNA_train.index.values,'miRNA':pred_miRNA, 'gene':pred_gene, 'meth':pred_meth, 'CNV':pred_CNV}
    # # new_feats = {'sample':miRNA_train.index.values,'gene':pred_gene, 'CNV':pred_CNV, 'meth':pred_meth}
    # new_feats = pd.DataFrame(data=new_feats)
    # new_feats = new_feats.set_index('sample')
    # # # print(new_feats)
    # # print(new_feats.head())
    # #
    # #
    # new_feats.to_csv('newfts1.csv')
    #
    # clf = tr_comb(new_feats,train_class_copy5)
    #
    # # validation
    # miRNA_test = miRNA_test[fea_miRNA]
    # gene_test = gene_test[fea_gene]
    # meth_test = meth_test[fea_meth]
    # CNV_test = CNV_test[fea_CNV]
    #
    #
    # pred_miRNA = clf_miRNA.decision_function(miRNA_test)
    # pred_gene = clf_gene.decision_function(gene_test)
    # pred_meth = clf_meth.decision_function(meth_test)
    # pred_CNV = clf_CNV.decision_function(CNV_test)
    #
    # # gets results from predicting on validation set with individual modalities
    # gene_ind_res = clf_gene.score(gene_test,test_class)
    # meth_ind_res = clf_meth.score(meth_test,test_class)
    # CNV_ind_res = clf_CNV.score(CNV_test,test_class)
    # miRNA_ind_res = clf_miRNA.score(miRNA_test,test_class)
    #
    # c1_gene, c2_gene, _ = roc_curve(test_class.values.ravel(), pred_gene.ravel())
    # c1_miRNA, c2_miRNA, _ = roc_curve(test_class.values.ravel(), pred_miRNA.ravel())
    # c1_CNV, c2_CNV, _ = roc_curve(test_class.values.ravel(), pred_CNV.ravel())
    # c1_meth, c2_meth, _ = roc_curve(test_class.values.ravel(), pred_meth.ravel())
    #
    # area_gene = auc(c1_gene, c2_gene)
    # area_miRNA = auc(c1_miRNA, c2_miRNA)
    # area_CNV = auc(c1_CNV, c2_CNV)
    # area_meth = auc(c1_meth, c2_meth)
    #
    #
    # new_feats_val = {'sample': miRNA_test.index.values, 'miRNA': pred_miRNA, 'gene': pred_gene, 'meth': pred_meth,
    #              'CNV': pred_CNV}
    # # new_feats_val = {'sample': miRNA_test.index.values,'gene': pred_gene,'CNV': pred_CNV,'meth':pred_meth}
    # new_feats_val = pd.DataFrame(data=new_feats_val)
    # new_feats_val = new_feats_val.set_index('sample')
    #
    # new_feats_val.to_csv('newftsval1.csv')
    #
    #
    # # weight_vote(new_feats_val,test_class,[miRNA_ind_res,gene_ind_res,meth_ind_res,CNV_ind_res])
    #
    #
    # fin = clf.score(new_feats_val,test_class)
    # pred = clf.decision_function(new_feats_val)
    # c1, c2, _ = roc_curve(test_class.values.ravel(), pred.ravel())
    # area = auc(c1, c2)
    # # fin = maj_vote(new_feats_val,test_class)
    # print('gene individual result:',gene_ind_res)
    # print('meth individual result:',meth_ind_res)
    # print('CNV individual result:',CNV_ind_res)
    # print('miRNA individual result:',miRNA_ind_res)
    # print('auc',area)
    # print('integration result',fin)





























    # ### BAR GRAPH COMPARING INTEGRATION OF DIFF MODALITY COMBOS ### - must comment out above integration
    #
    # fins = []
    # areas = []
    # fin1s = []
    # combinations = [["meth"],["miRNA"],["gene"],["CNV"],["meth","miRNA"],["meth","gene"],["meth","CNV"],
    #                 ["miRNA","gene"],["miRNA","CNV"],["gene","CNV"],["meth","miRNA","gene"],
    #                 ["meth","miRNA","CNV"],["miRNA","gene","CNV"],
    #                 ["meth","gene","CNV"],["meth","miRNA","gene","CNV"]]
    # # select features
    # miRNA_train_copy2 = miRNA_train_copy2[fea_miRNA]
    # gene_train_copy2 = gene_train_copy2[fea_gene]
    # meth_train_copy2 = meth_train_copy2[fea_meth]
    # CNV_train_copy2 = CNV_train_copy2[fea_CNV]
    # # gene_test = gene_test[fea_gene]
    # #
    # pred_miRNA = clf_miRNA.decision_function(miRNA_train_copy2)
    # pred_gene = clf_gene.decision_function(gene_train_copy2)
    # pred_meth = clf_meth.decision_function(meth_train_copy2)
    # pred_CNV = clf_CNV.decision_function(CNV_train_copy2)
    # # pred_gene = clf_gene.decision_function(gene_test)
    # # print(pred_gene)
    # # print(clf_gene.predict(gene_test))
    # # c1,c2,_ = roc_curve(test_class.values.ravel(), pred_gene.ravel())
    # # print(auc(c1, c2))
    # # print(clf_gene.score(gene_test,test_class))
    # #
    # new_feats = {'sample': miRNA_train.index.values, 'miRNA': pred_miRNA, 'gene': pred_gene, 'meth': pred_meth,
    #              'CNV': pred_CNV}
    # new_feats = pd.DataFrame(data=new_feats)
    # new_feats = new_feats.set_index('sample')
    # # print(new_feats)
    # # new_feats.to_csv('new_feats.csv')
    # # train_class_copy5.to_csv('new_feats_labels.csv')
    # print(new_feats.head())
    #
    # for com in combinations:
    #     print(com)
    #     clf = tr_comb(new_feats[com], train_class_copy5)
    #
    #     # validation
    #     miRNA_test = miRNA_test[fea_miRNA]
    #     gene_test = gene_test[fea_gene]
    #     meth_test = meth_test[fea_meth]
    #     CNV_test = CNV_test[fea_CNV]
    #
    #     pred_miRNA = clf_miRNA.decision_function(miRNA_test)
    #     pred_gene = clf_gene.decision_function(gene_test)
    #     pred_meth = clf_meth.decision_function(meth_test)
    #     pred_CNV = clf_CNV.decision_function(CNV_test)
    #
    #     fin1 = clf_gene.score(gene_test, test_class)
    #
    #     new_feats_val = {'sample': miRNA_test.index.values, 'miRNA': pred_miRNA, 'gene': pred_gene, 'meth': pred_meth,
    #                      'CNV': pred_CNV}
    #     new_feats_val = pd.DataFrame(data=new_feats_val)
    #     new_feats_val = new_feats_val.set_index('sample')
    #     # print(new_feats_val.head())
    #
    #     fin = clf.score(new_feats_val[com], test_class)
    #     pred = clf.decision_function(new_feats_val[com])
    #     c1, c2, _ = roc_curve(test_class.values.ravel(), pred.ravel())
    #     area = auc(c1, c2)
    #     # print(fin1)
    #     print(area)
    #     print(fin)
    #     fin1s.append(fin1)
    #     areas.append(area)
    #     fins.append(fin)
    # # print(fin1s)
    #
    # fins[0] = meth_ind_res
    # fins[1] = miRNA_ind_res
    # fins[2] = gene_ind_res
    # fins[3] = CNV_ind_res
    # areas[0] = area_meth
    # areas[1] = area_miRNA
    # areas[2] = area_gene
    # areas[3] = area_CNV
    #
    # print(areas)
    # print(fins)
    #
    # n_groups = 15
    #
    # fig, ax = plt.subplots()
    #
    # index = np.arange(n_groups)
    # bar_width = 0.35
    #
    # opacity = 0.4
    #
    # # rects1 = ax.bar(index, fin1s, bar_width,
    # #                 alpha=opacity, color='b',
    # #                 label='fin1')
    #
    # rects2 = ax.bar(index, areas, bar_width,
    #                 alpha=opacity, color='r',
    #                 label='auc')
    #
    # rects3 = ax.bar(index + bar_width, fins, bar_width,
    #                 alpha=opacity, color='b',
    #                 label='score')
    #
    # ax.set_xlabel('Combination')
    # ax.set_ylabel('Scores')
    #
    # ax.set_xticks(index + bar_width / 2)
    # ax.set_xticklabels(combinations)
    # ax.legend()
    #
    # fig.tight_layout()
    # plt.show()
    #
    # ### END ###

pipeline(True)






