#!/usr/bin/env python
#%%
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

#%% multi-label
fn = "../res200225.csv"
dat = pd.read_csv(fn, header=0)
th = 0.5
t = dat['true L'].values   # true score
pred = dat['pred L'].values # prediction score
#pred[dat['pred max e']>0.5]=0
#pred[dat['pred max 4']>0.3]=0 # if class 4
print("rows: prediction, cols: truth")
print(confusion_matrix(t, np.round(pred)))
print(classification_report(t, np.round(pred)))
#print(pred)
#  evaluation score
score = 7 * (np.abs(t-pred) > 2.5).sum()
score += 2 * (np.abs(t-pred) > 1.5).sum()
score += 1 * (np.abs(t-pred) > 0.5).sum()
print("avg score: ",score/len(t))
#%%
miss = (np.abs(t-pred) > 1.5) #2.5
for i in range(4):
    m = ((t == i) & miss).sum()
    total = (t==i).sum()
    print("t={}: {} / {}, ratio {:5f}".format(i,m,total,m/total))
print("total: {} / {}, ratio {:5f}".format(miss.sum(),len(t),miss.sum()/len(t)))
#%%
#master = pd.read_csv("../all20200221.csv", header=0)
master = pd.read_csv("../all200128.csv", header=0)
for k in master.keys()[1:]:
    print(k, sum(master[k]==1), sum(master[k]==2), sum(master[k]==3))

#%%
th = 1.5
print(confusion_matrix(t>=2, pred>th))
print(classification_report(t>=2, pred>th))
fp, tp, th = roc_curve(t>=2,pred)
plt.plot(fp, tp, marker='o')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid()
plt.show()

print("AUC: {}".format(roc_auc_score(t>0,pred)))
print("FP rate for 90% recall",fp[np.min(np.where(tp > 0.9))])
#%%

fp, tp, th = roc_curve(t>0,pred)
plt.plot(fp, tp, marker='o')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.grid()
plt.show()

print("AUC: {}".format(roc_auc_score(t>0,pred)))

#%%  4 class
fn = "result/result_4.csv"
dat = pd.read_csv(fn, header=None)
t = dat.iloc[:,1].values   # true label
pred = dat.iloc[:,2].values   # true label
prob = dat.iloc[:,3:7].values # prediction score
print("{}".format([np.sum(t==i) for i in range(4)]))

#%%
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
print(confusion_matrix(t,pred))
print("Accuracy {}, F1 {}".format(accuracy_score(t,pred),f1_score(t,pred,average="micro")))

#%% mis-classified data
fn = "result/result_4.csv"
dat = pd.read_csv(fn, header=None)
master = pd.read_csv("all_new.csv", header=0)
df = pd.DataFrame()
p = np.where(dat.iloc[:,1].values != dat.iloc[:,2].values)[0]

for i in p:
    f = dat.iloc[i,0]
    line = master[master['filename'] == f]
    line['true'] = dat.iloc[i,1]
    line['pred'] = dat.iloc[i,2]
    df = df.append(line,ignore_index=True)
df.to_csv("misclassified.csv", header=True, index=False)
#%%
#%%
import shutil
dat = pd.read_csv("misclassified.csv", header=0)
for f in dat.iloc[:,1].values:
    print(f)
    shutil.copyfile("image/"+f,"relimg/"+f)


#%%
dat

#%%
