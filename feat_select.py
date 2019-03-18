import pandas as pd
import pymrmr
# import mifs
# import mrmr

def select_features(X,y,modality):
    #combine response with features to one dataframe [y,X]
    z = pd.concat([y,X],axis=1)
    z = discretize(z,modality)

    # return (features)

def discretize(df,modality): #features need to be -2,0,2 and response needs to be -1,1
    if modality == 'CNV':
        df[df == -1] = -2
        df[df == 0] = 0
        df[df == 1] = 2
    else:
        # get mean and standard deviation each row column (since features are now columns)
        std = df.std(axis=0)
        av = df.mean(axis=0)

        df[df < av-std] = -2
        df[df > av+std] = 2
        df[df > av-std and df < av+std] = 0



# # initialize list of lists
# data1 = [[5, 5], [5, 5], [0, 0]]
# data2 = [[0, 0], [0, 0], [1, 1]]
#
# # Create the pandas DataFrame
# df1 = pd.DataFrame(data1, columns = ['c1', 'c2'])
# df2 = pd.DataFrame(data2, columns = ['c1', 'c2'])
# print(df2)
# cool = df2.std(axis=0)
# cool2 = df2.mean(axis=0)
# print(cool2)
