import pandas as pd
import numpy as np
import csv
import os

# way to index a df df.iloc[0,0])
# way to get col names list(df)

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

# normalizing gene expression and miRNA datasets
gene = gene.div(gene.max(axis=1), axis=0)
miRNA = miRNA.div(miRNA.max(axis=1), axis=0)

# splitting labels into train set and validation set








