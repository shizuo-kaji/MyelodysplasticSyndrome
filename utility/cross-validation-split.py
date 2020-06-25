#!/usr/bin/env python
#%%
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import KFold
#from skmultilearn.model_selection import iterative_train_test_split

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('filename', help='csv file')
args = parser.parse_args()

#%%
dat = pd.read_csv(args.filename, header=0)
for i in range(4):
    print( "Label {}:".format(i), (dat.iloc[:,1:] == i).sum(axis=0))

#%% stratified split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)
i=0
for train_index, test_index in mskf.split(dat,dat.iloc[:,1:]):
    dat.iloc[train_index].to_csv("cv/train_cv{}.csv".format(i), header=True, index=False)
    dat.iloc[test_index].to_csv("cv/test_cv{}.csv".format(i), header=True, index=False)
#    print(dat.iloc[test_index,1:].sum())
    print(np.sum(dat.iloc[test_index,1:].values,axis=0))
    i += 1

#%% simple split
def simple_split(dat):
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    i=0
    for train_index, test_index in cv.split(dat):
    #    print(train_index,test_index)
        dat.iloc[train_index].to_csv("train_cv{}.csv".format(i), header=True, index=False)
        dat.iloc[test_index].to_csv("test_cv{}.csv".format(i), header=True, index=False)
    #    print(dat.iloc[test_index,2:].sum())
        print(np.sum(dat.iloc[test_index,2:].values,axis=0))
        i += 1

#%%
def proba_mass_split(y, folds=5):
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    print("Fold distributions are")
    print(fold_dist)
    return index_list

#p=np.random.permutation(len(dat))
#L=proba_mass_split(dat.values[p,2:].astype(int))

#%%
