#!/usr/bin/env python
## Usage: python excel2label.py F3F11crop.xlsx > F3F11crop.csv

import sys
import os
import operator
from collections import defaultdict
import pandas as pd

import argparse
import random
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('path', help='excel file')
parser.add_argument('--skip', default=1, type=int, help='skip rows')
args = parser.parse_args()

#A,B,C,D,E,J,L,P,Q,R,T,U,X => 0,1,2,3 を取る
#Z,0e,0g,0m,対象外,評価不能 => 0,1 を取る 
# 4 は細分された

labels = ['0e','A','B','C','D','E','G','X','Y','giant','0g','J','K','L','P','Q','R','Z','0m','S','T','U','V','img','pla','lym','mei','eos','his','mon','bas','div','rbc','plt']
labels_flag =['Z','0e','0g','0m','img','pla','lym','mei','eos','his','mon','bas','div','rbc','plt']
# load excel file
df = pd.read_excel(args.path,header=None,skiprows=args.skip)

print("Filename,"+",".join(labels))
for key,line in df.iterrows():
#    print(line.iat[0])
    vec = ["0"] * len(labels)
    for i in range(1,len(line)):
        if str(line.iat[i]) in labels_flag:
            vec[labels.index(str(line.iat[i]))] = "1"
        elif line.iat[i] in labels:
            vec[labels.index(line.iat[i])] = str(line.iat[i+1])
            i += 1
    print(line.iat[0]+","+",".join(vec))
#    if vec == ["0"] * len(labels):
#        print(line)
