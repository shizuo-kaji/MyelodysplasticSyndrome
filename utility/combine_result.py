#!/usr/bin/env python
#%% Usage: python excel2label.py F3F11crop.xlsx > F3F11crop.csv

import sys
import os
import operator
from collections import defaultdict
import pandas as pd

import argparse
import random
import sys

#%%
parser = argparse.ArgumentParser(description='')
parser.add_argument('--master', '-m', help='master excel file',default="200221whole.xlsx")
parser.add_argument('--result', '-r', help='result csv file',default="result200117.csv")
args = parser.parse_args()

master = pd.read_excel(args.master,header=0, index_col=0)
res = pd.read_csv(args.result,header=0, index_col=0)

for key,line in res.iterrows():
    if key=="filename":
        continue
    L = [key]+line.to_list()+master.loc[key].to_list()
    for a in L:
        print(a,end=",")
    print()
#    print(line+","+",".join(mast)er.loc[key]))


# %%
