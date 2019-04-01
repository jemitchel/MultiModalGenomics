import pandas as pd
import os
from sklearn.model_selection import train_test_split
from train_ind import tr_ind
from sklearn import preprocessing
from feat_select import select_features
from train_comb import tr_comb

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
    miRNA_scaler = preprocessing.RobustScaler().fit(miRNA_train)
    miRNA_train = miRNA_scaler.transform(miRNA_train)
    miRNA_train = pd.DataFrame(miRNA_train,columns=list(miRNA_train_copy)).set_index(miRNA_train_copy.index.values)

    gene_train_copy = pd.DataFrame(gene_train, copy=True) # copies the original dataframe
    gene_scaler = preprocessing.RobustScaler().fit(gene_train)
    gene_train = gene_scaler.transform(gene_train)
    gene_train = pd.DataFrame(gene_train,columns=list(gene_train_copy)).set_index(gene_train_copy.index.values)

    train_class = train_class.set_index('case_id') # changes first column to be indices
    test_class = test_class.set_index('case_id') # changes first column to be indices

    # makes copies of the y dataframe because tr_ind alters it
    train_class_copy1,train_class_copy2,train_class_copy3,train_class_copy4,train_class_copy5 = pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True),\
                                                                                 pd.DataFrame(train_class, copy=True)
    gene_train_copy2 = pd.DataFrame(gene_train, copy=True)
    miRNA_train_copy2 = pd.DataFrame(miRNA_train, copy=True)
    meth_train_copy2 = pd.DataFrame(meth_train, copy=True)
    CNV_train_copy2 = pd.DataFrame(CNV_train, copy=True)

    # do cross validation to get best classifiers and feature sets for each modality
    clf_gene, fea_gene = tr_ind(gene_train,train_class_copy1,'gene','mrmr')
    clf_miRNA, fea_miRNA = tr_ind(miRNA_train,train_class_copy2,'miRNA','mrmr')
    clf_meth, fea_meth = tr_ind(meth_train,train_class_copy3,'meth','mrmr')
    clf_CNV, fea_CNV = tr_ind(CNV_train,train_class_copy4,'CNV','chi-squared')

    # select features
    miRNA_train_copy2 = miRNA_train_copy2[fea_miRNA]
    gene_train_copy2 = gene_train_copy2[fea_gene]
    meth_train_copy2 = meth_train_copy2[fea_meth]
    CNV_train_copy2 = CNV_train_copy2[fea_CNV]

    pred_miRNA = clf_miRNA.decision_function(miRNA_train_copy2)
    pred_gene = clf_gene.decision_function(gene_train_copy2)
    pred_meth = clf_meth.decision_function(meth_train_copy2)
    pred_CNV = clf_CNV.decision_function(CNV_train_copy2)

    new_feats = {'sample':miRNA_train.index.values,'miRNA':pred_miRNA, 'gene':pred_gene, 'meth':pred_meth, 'CNV':pred_CNV}
    new_feats = pd.DataFrame(data=new_feats)
    new_feats = new_feats.set_index('sample')
    print(new_feats)
    new_feats.to_csv('new_feats.csv')

    tr_comb(new_feats,train_class_copy5)



pipeline(True)






