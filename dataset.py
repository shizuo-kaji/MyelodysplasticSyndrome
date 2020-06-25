#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import functools
import glob
import os
import random
import shutil
from datetime import datetime as dt

import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib as mpl
import numpy as np
import pandas as pd
from chainer import datasets, iterators, training
from chainer.dataset import concat_examples, convert, dataset_mixin
from chainer.training import extensions, triggers

import chainercv
import cv2
import losses
from chainercv2.model_provider import get_model as chcv2_get_model
from chainercv.transforms import (center_crop, random_crop, random_flip,
                                  resize, resize_contain, rotate)
from chainercv.utils import read_image, write_image
#from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from consts import dtypes, optim, columns

THRESHOLD = 0.5

## dataset preparation
class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, args, df, augment, random=0):
        self.path = args.root
        self.ids = []  # filenames
        self.random = random
        self.dup = 1
        self.scaling_jitter = args.scaling_jitter
        self.arch = args.arch
        self.local_average_subtraction = args.local_average_subtraction
        self.maxval = np.array([1]+[3]*9 + [1]+[3]*7 + [1]+[3]*4 + [1]*11)

        # dir for preprocessed images to be saved
        if args.save_preprocessed_image:
            self.save_preprocessed_image=os.path.join(args.outdir,args.save_preprocessed_image)
        else:
            self.save_preprocessed_image=None
        self.mean = np.array([103.063, 115.903, 123.152],dtype=np.float32)[:,np.newaxis,np.newaxis]  # mean (B,G,R)
        if args.bg_image:
            self.bg_image = cv2.imread(args.bg_image)
        else:
            self.bg_image = np.zeros((1024,1024,3))
        colname = df.columns.tolist()[1:]   # 0 is filename
        # augment relevant rows
        M = self.maxval[args.target_id]
        dd = [df[df.iloc[:,args.target_id+1] == i] for i in range(M+1)]
        print("#Samples in each cls:", [len(d) for d in dd]) ## num of samples for each abnormal level
        if augment=='equal': # how many times each sample is fed
            for i in range(1,M+1):
                for j in range( int(len(dd[0])/len(dd[i]))-1 ):
                    df = df.append(dd[i],ignore_index=True)
        elif augment.isnumeric():
            self.dup = int(augment)
            df = pd.concat([df]*self.dup)
        df.sort_values(['L','Filename'], ascending=False, inplace=True)
        self.ids = df.iloc[:,0].values  # image filename
        ## 
        dat = df.iloc[:,1:35].values.astype(np.int64)

        print("Augmented to: ",[len(dat[dat[:,args.target_id] == i]) for i in range(M+1)])
        # crop size (input size to pretrained CNN)
        if self.arch in ['inceptionv4','xception','efficientnet_b4b']:
            self.ch=299
        else:
            self.ch=224
        if not args.scale_to:
            self.scale_to = int(self.ch*5/3)  ## cut 1/5 from every edge
        else:
            self.scale_to = args.scale_to
        if args.case == '4': ## 4-class classification
            if not (np.max(dat,axis=1)>0).all():
                print("some labels are wrong!")
                exit()
            u = np.argmax(dat,axis=1) 
            self.cls = np.zeros(len(dat),dtype=np.int64)
            self.cls[u>9]=1
            self.cls[u>17]=2
            self.cls[u>22]=3
            self.chs = np.max(self.cls) + 1 ## number of classes
            self.header = "filename,true,pred,score"
            print("#class {}".format([sum(self.cls==i) for i in range(self.chs)]))
        else:
            if args.case in colname:
                i = colname.index(args.case)
                m = (i,i+1)
            else:
                m = columns[args.case]
            if args.target < 0:
                targetcol = m
            else:
                targetcol = slice(args.target,args.target+1)
            if args.regression:
                self.cls = dat[:,targetcol].astype(np.float32) ## regression for each label (single-label)
            else:
                self.cls = (dat[:,targetcol] > 0).astype(np.int64) ## yes/no for each label (multi-label)
            self.chs = self.cls.shape[1] # output dimension
            self.header = ["filename","true L","pred L","score","average L","min L","max L","true max 4","pred max 4","true max e","pred max e","true max m","pred max m"]
            self.header += [f+"_true,prob" for f in colname[targetcol]]
            self.header = ",".join(self.header)
            print("Predicting: ",colname[targetcol])
            print("#Positive samples: ",np.sum(self.cls>0,axis=0))
            print("#Total samples: {}".format(len(self.cls)))
#            for i in range(len(colname)):
#                print(colname[i],np.max(self.cls[:,i]))
    
        if self.save_preprocessed_image:
            os.makedirs(self.save_preprocessed_image, exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def get_img_path(self, i):
        return os.path.join(self.path,self.ids[i])

    def get_example(self, i):
        img = self.bg_image.copy()
        H,W,_ = img.shape
        img_in = cv2.imread(self.get_img_path(i)) # BGR, HWC
        H0,W0,_=img_in.shape
        scale = self.scale_to / max(H0,W0)
        if self.random>0:
            scale *= random.uniform(1.0-self.scaling_jitter,1.0+self.scaling_jitter)
        if scale>0:
            img_in = cv2.resize(img_in, dsize=None, fx=scale, fy=scale)
        oH,oW,_ = img_in.shape
        y_offset = int(round((H - oH) / 2.))
        x_offset = int(round((W - oW) / 2.))
        y_slice = slice(y_offset, y_offset + oH)
        x_slice = slice(x_offset, x_offset + oW)
        img[y_slice, x_slice,:] = img_in # paste onto background image
        img = np.transpose(img,(2,0,1)) # [C,H,W]
        if self.save_preprocessed_image:
            write_image(img[::-1,:,:],os.path.join(self.save_preprocessed_image,"f_"+self.ids[i]))
#        img = resize_contain(img, size=(self.scale_to,self.scale_to), fill=self.mean)
        if self.random>0:
            img = random_flip(img, x_random=True, y_random=False)
            img = rotate(img, random.uniform(-180,180), expand=False)
        else:
            img = rotate(img, (360/self.dup)*i,expand=False)
#            print(self.dup, (360/self.dup)*i)
        if self.local_average_subtraction>0:
            img = (img-cv2.blur(img, ksize=(self.local_average_subtraction, self.local_average_subtraction))+128.0)
        img = center_crop(img,(self.ch+self.random, self.ch+self.random))
        if self.random>0:
            img = random_crop(img, (self.ch,self.ch))
        if self.save_preprocessed_image:
            write_image(img[::-1,:,:],os.path.join(self.save_preprocessed_image,"{}_".format(i)+self.ids[i]))
        img = img.astype(np.float32)-self.mean
        return (img,self.cls[i])
