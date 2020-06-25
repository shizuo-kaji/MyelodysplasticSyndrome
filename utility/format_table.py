#! /usr/local/bin/python
# 
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
test = pd.read_csv("test.csv", header=0)
pred = np.load("predict.npy")
#%%
ks = test.keys()[2:]
tps = []
fns = []
fps = []
for i,x in enumerate(test.iterrows()):
    tp = ""
    fp = ""
    fn = ""
    for j,k in enumerate(ks):
        if x[1][k]>0:
            if pred[i,j]>0.5:
                tp = tp + k[:2] +"{0:1.0f} ".format(10*(pred[i,j]-0.001))
            else:
                fn = fn + k[:2] +"{0:1.0f} ".format(10*pred[i,j])
        elif pred[i,j]>0.5:
            fp = fp + k[:2] +"{0:1.0f} ".format(10*(pred[i,j]-0.001))
    tps.append(tp)
    fns.append(fn)
    fps.append(fp)
#%%
test["TP"] = tps
test["FN"] = fns
test["FP"] = fps
#%%
test.to_csv("out.csv")
#%%

#%%
