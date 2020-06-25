import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainercv
import cv2
import matplotlib as mpl
import numpy as np
import pandas as pd
from chainer import datasets, iterators, training
from chainer.dataset import concat_examples, convert, dataset_mixin
from chainer.training import extensions, triggers
from chainercv2.model_provider import get_model as chcv2_get_model

from consts import dtypes, optim
import os
from datetime import datetime as dt
from chainerui.utils import save_args
import argparse


## identity link
class IdentityLayer(chainer.Link):
    def __init__(self):
        super(IdentityLayer, self).__init__()
    def forward(self, x):
        return F.identity(x)

## NN definition
class CNN(chainer.Chain):
    def __init__(self, args, maxval=None):
        self.dropout = args.dropout
        self.layer = args.layer
        self.fch = args.fch
        self.arch = args.arch
        self.maxval = maxval
        self.global_pool = args.global_pool
        super(CNN, self).__init__()
        with self.init_scope():
            if args.arch == "senext":
                self.base = chainercv.links.model.senet.SEResNeXt101(pretrained_model='imagenet',mean=np.array([0,0,0]))
                self.base.pick = args.layer
            elif args.arch == "seres":
                self.base = chainercv.links.model.senet.SEResNet152(pretrained_model='imagenet',mean=np.array([0,0,0]))
                self.base.pick = args.layer
            elif args.arch == "resnet152":
                self.base = L.ResNet152Layers()
            elif args.arch == "resnet152_nopretrain": 
                self.base = L.ResNet152Layers(pretrained_model=None)
            elif args.arch == "resnet101":
                self.base = L.ResNet101Layers()
            elif args.arch == "resnet50":
                self.base = L.ResNet50Layers()
            else:  # chainercv2
                self.base = chcv2_get_model(args.arch, pretrained=True)
                self.base.output=IdentityLayer()
#            pointwise = L.Convolution2D(None, len(args.cols), 1, 1, 0, initialW=w, initial_bias=bias),
        ## add fc layers for finetuning
            for i in range(len(args.fch)):
                setattr(self, 'fc' + str(i), L.Linear(None,args.fch[i]))
            self.fcl = L.Linear(None, args.chs)
    def __call__(self, x):
        ## feature extraction
        if "resnet" in self.arch:
            h = self.base(x, layers=[self.layer])[self.layer]
        else:
            h = self.base(x)
        ## final output layer
        h = F.dropout(h,ratio=self.dropout)
#        print(h.shape)
        if self.global_pool == "average":
            h = F.average_pooling_2d(h, h.shape[2:4])
        elif self.global_pool == "max":
            h = F.max_pooling_2d(h, h.shape[2:4])
        for i in range(len(self.fch)):
            h = F.relu(getattr(self, 'fc' + str(i))(h))
            h = F.dropout(h,ratio=self.dropout)
        h = self.fcl(h)
        if self.maxval is not None:
            h = self.xp.asarray(self.maxval)*F.sigmoid(h)
        return h
