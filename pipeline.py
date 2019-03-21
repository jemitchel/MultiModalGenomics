import pandas as pd
import os
from sklearn.model_selection import train_test_split
from train_ind import tr_ind
from sklearn import preprocessing
import numpy as np

os.chdir("C:\\Users\\jonat\\Documents\\Spring 2019 Classes\\4813\\BHI_data")
labels = pd.read_csv('Label_selected.csv')

ndx = labels.index[labels['label'] > -1].tolist() #gets indices of patients to use
lbl = labels.iloc[ndx,[0,4]] #makes a new dataframe of patients to use (their IDs and survival response)

# miRNA dataset
miRNA = pd.read_csv('miRNA_selected.csv') #reads in dataset
miRNA = miRNA.set_index('miRNA_ID') #changes first column (miRNA_ID) to be indices of the dataframe
miRNA = miRNA[lbl['case_id']] #indexes out the correct patient samples

# gene expression dataset
gene = pd.read_csv('GeneExp_selected.csv') #reads in dataset
gene = gene.set_index(gene.columns[0]) #changes first column to be indices of the dataframe
gene = gene[lbl['case_id']] #indexes out the correct patient samples

# CNV dataset
CNV = pd.read_csv('CNV_selected.csv') #reads in dataset
CNV = CNV.set_index('Gene_Symbol') #changes first column to be indices of the dataframe
CNV = CNV[lbl['case_id']] #indexes out the correct patient samples

# DNA methylation dataset
meth = pd.read_csv('DnaMeth_selected.csv') #reads in dataset
meth = meth.set_index('Composite Element REF') #changes first column to be indices of the dataframe
meth = meth[lbl['case_id']] #indexes out the correct patient samples

# removes rows (features) that are all 0 across patients
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


# # normalizing gene expression and miRNA datasets
# gene = gene.div(gene.max(axis=1), axis=0)
# miRNA = miRNA.div(miRNA.max(axis=1), axis=0)

miRNA_train_copy = pd.DataFrame(miRNA_train, copy=True) # copies the original dataframe
miRNA_scaler = preprocessing.RobustScaler().fit(miRNA_train)
miRNA_train = miRNA_scaler.transform(miRNA_train)
miRNA_train = pd.DataFrame(miRNA_train,columns=list(miRNA_train_copy)).set_index(miRNA_train_copy.index.values)

gene_train_copy = pd.DataFrame(gene_train, copy=True) # copies the original dataframe
gene_scaler = preprocessing.RobustScaler().fit(gene_train)
gene_train = gene_scaler.transform(gene_train)
gene_train = pd.DataFrame(gene_train,columns=list(gene_train_copy)).set_index(gene_train_copy.index.values)

train_class = train_class.set_index('case_id') # changes first column to be indices

tr_ind(miRNA_train,train_class,'miRNA') # fs + classification cv






