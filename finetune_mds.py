#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

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
#from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from functools import partial
import losses
from consts import dtypes, optim, columns
from dataset import Dataset
from arguments import arguments
from pretrained_net import CNN

## global configuration
mpl.use('Agg')
THRESHOLD = 0.5

## accuracy in terms of positive or not
def pos_nopos(x,t,target_id):
    y = x.array
    if target_id>-1:
        ym = y[:,target_id]
        ym[ym < x.xp.max(y[:,columns['o']],axis=1)] = 0
        ym[ym < x.xp.max(y[:,columns['m']],axis=1)] = 0
        ym[ym < x.xp.max(y[:,columns['e']],axis=1)] = 0
        return(x.xp.mean(x.xp.abs(ym-t[:,target_id])<1.5))
#        return(x.xp.mean((ym>THRESHOLD) == (t[:,target_id]>THRESHOLD)))
    else:
        return(x.xp.mean((y>THRESHOLD) == (t>THRESHOLD)))

## weighted norm
def weighted_norm(x,t, target_id, label_weight=10, exponent=4):
    w = x.xp.ones_like(t)
#    w[:,target_id] = (t[:,target_id]+1)**2
    w[:,target_id] *= label_weight
    return( F.average(w * ((x-t)**exponent) )
)

## score (TODO: other than L)
def L_score(x,t,target_id):
    y = x.array
    tm = t[:,target_id] # int
    ym = y[:,target_id] # 
    ym[ym < x.xp.max(y[:,columns['o']],axis=1)] = 0
    ym[ym < x.xp.max(y[:,columns['m']],axis=1)] = 0
    ym[ym < x.xp.max(y[:,columns['e']],axis=1)] = 0
    sc =  7 * ((x.xp.abs(tm-ym) > 2.5).sum())
    sc += 2 * ((x.xp.abs(tm-ym) > 1.5).sum())
    sc += 1 * ((x.xp.abs(tm-ym) > 0.5).sum())
    return(sc/len(t))

# prediction and statistics (TODO: other than L)
def pred(test,args,model,filename,output_all=True):
    test_iter = iterators.MultithreadIterator(test, test.dup, repeat=False, shuffle=False, n_threads=args.loaderjob)  # each batch contains random variations of a single example
    idx=0
    fp,tp,fn = 0,0,0
    with open(os.path.join(args.outdir,filename),'w') as output:
        output.write(test.header+"\n")
        score = 0
        for batch in test_iter:
            x, t = concat_examples(batch, device=args.gpu)
            with chainer.using_config('train', False):
                with chainer.function.no_backprop_mode():
                    if args.regression:
                        y = model.predictor(x).data # regression
                    else:
                        y = F.softmax(model.predictor(x)).data  # classification
            if args.gpu>-1:
                y = model.xp.asnumpy(y)
                t = model.xp.asnumpy(t)            
            # output to file
            for i in range(len(y)):
                fname = os.path.basename(test.ids[idx])
#                 output.write(fname)
#                 if(len(t.shape)>1): ## multi-label
#                     for j in range(t.shape[1]):
#                         output.write(",{:.0f}".format(t[i,j]))
# #                        output.write(",{}".format(y[i,j]>=0))  # classification with linear output
#                         output.write(",{:.0f}".format(np.round(y[i,j])))
#                         output.write(",{}".format(y[i,j]))
#                 else: ## single label
#                     output.write(",{:.0f}".format(t[i]))
#                     output.write(",{:.0f}".format(np.argmax(y[i,:])))
#                     for yy in y[i]:
#                         output.write(",{0:1.5f}".format(yy))
#                 output.write("\n")
                idx += 1
            # statistics for each file
            if t.shape[1]==34: # ALL
                ym = y[:,args.target_id] # float[test.dup]
                tm = t[0,args.target_id] # int
                sc_y = np.max(ym)
                sc_o = np.max(y[:,columns['o']])
#                sc_o = np.any( (y[:,columns['o']]) >THRESHOLD)  # bool
                sc_m = np.max(y[:,columns['m']]) 
                sc_e = np.max(y[:,columns['e']]) 
                yfinal = 0 if max(sc_o,sc_m,sc_e)>sc_y else sc_y
                tr = tm>THRESHOLD
                pred = yfinal > THRESHOLD
                # evaluation score
                sc = 7 * ((np.abs(tm-yfinal) > 2.5).sum())
                sc += 2 *  ((np.abs(tm-yfinal) > 1.5).sum())
                sc += 1 *  ((np.abs(tm-yfinal) > 0.5).sum())
                score += sc
                # output
                if pred != tr or output_all:
                    # ["filename","true L","pred L","score","average L","min L","max L","true 4","max 4","true e","max e"]
                    output.write("{},{:.0f},{:.0f},{:.0f},{:.5f},{:.5f},{:.5f}".format(fname,tm,yfinal,sc,np.mean(ym),np.min(ym),np.max(ym)))
                    output.write(",{:.0f},{:.5f},{:.0f},{:.5f},{:.0f},{:.5f}".format(np.max(t[0,columns['o']]),sc_o,np.max(t[0,columns['e']]),sc_e,np.max(t[0,columns['m']]),sc_m))
                    for i in range(y.shape[1]):
                        output.write(",{:.0f},{:.5f}".format(t[0,i],np.max(y[:,i])))
                    output.write("\n")
            else:
                pred = (y>THRESHOLD)
                tr = (t>THRESHOLD)
            #
            tp += np.sum(pred == tr)
            fn += np.sum(np.logical_and(np.logical_not(pred),tr))
            fp += np.sum(np.logical_and(pred,np.logical_not(tr)))
    score /= (idx/test.dup)
    ac = 1-((fn+fp)/(idx/test.dup))
    return(ac,tp,fn,fp,score)

# evaluator
class Evaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        params = kwargs.pop('params')
        super(Evaluator, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.test = params['test']
        self.count = 0
    def evaluate(self):
        model = self.get_target('main')
        if self.eval_hook:
            self.eval_hook(self)
        filename = "result_{}.csv".format(self.count)
        ac,tp,fn,fp,score = pred(self.test,self.args,model,filename)
        with open(os.path.join(self.args.outdir,"summary_{}.txt".format(self.count)),"w") as fh:
            fh.write("Acc {}, TP {}, FN {}, FP {}, score {}".format(ac,tp,fn,fp,score))
        self.count += 1
        return {"myval/Acc":ac, "myval/FN":fn, "myval/FP":fp, "myval/score":score}

########################################################
def main():
    args = arguments()
    dtime = dt.now().strftime('%m%d_%H%M')
    args.outdir = os.path.join(args.outdir, '{}_mds_{}'.format(dtime,args.case))

    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]
    chainer.print_runtime_info()

    dirname = os.path.join("result",'{}'.format(dtime))
    # read training data from file
    if args.train[:2] == "cv" and args.train[2].isdigit():
        # if cv is specified
        i = args.train[2]
        args.train = "cv/train_cv{}.csv".format(i)
        args.val = "cv/test_cv{}.csv".format(i)
        args.outdir += "_cv{}".format(i)
        dirname += "_cv{}".format(i)
        print(args.train)
        tdf = pd.read_csv(args.train, header=0)
        vdf = pd.read_csv(args.val, header=0)
    else:
        # if filename is specified
        tdf = pd.read_csv(args.train, header=0)
        vdf = pd.read_csv(args.val, header=0)
    
    train = Dataset(args,tdf,augment=args.augment,random=args.random)
    test =  Dataset(args,vdf,augment=args.ensemble,random=0)
#    train = Dataset(args,tdat,augment=args.ensemble,random=0)   ## to get figures
        
    print(args)
    save_args(args, args.outdir)
    args.chs = train.chs
    print("Number of classes {}".format(args.chs))
    # Set up a neural network to train
    if args.case == "4":  # classification
        if args.loss=='contrastive':
            args.chs = 64 # dimension of the metric space
            model = L.Classifier(CNN(args), lossfun=losses.contrastive_loss, accfun=losses.contrastive_loss)
            model.compute_accuracy = False
        elif args.loss == 'focal':
            model = L.Classifier(CNN(args), lossfun=losses.softmax_focalloss, accfun=F.accuracy) 
        else:
            model = L.Classifier(CNN(args), lossfun=F.softmax_cross_entropy, accfun=F.accuracy)    
    else: # multi-label
        if args.loss == 'focal':
            model = L.Classifier(CNN(args), lossfun=losses.sigmoid_focalloss, accfun=F.binary_accuracy)
        elif args.loss == 'rmse':
#            train.maxval = None   ## TEST
            model = L.Classifier(CNN(args,maxval=train.maxval), lossfun=partial(weighted_norm,target_id=args.target_id,label_weight=args.class_weight,exponent=args.loss_exponent),
                                                                accfun=partial(pos_nopos,target_id=args.target_id))  #L_score
        else:
            model = L.Classifier(CNN(args), lossfun=F.sigmoid_cross_entropy, accfun=F.binary_accuracy)
    
    # Set up an optimizer
    optimizer = optim[args.optimizer](args.learning_rate)
    optimizer.setup(model)
    if args.weight_decay_l2>0:
        if args.optimizer in ['Adam','AdaBound','Eve']:
            optimizer.weight_decay_rate = args.weight_decay_l2
        else:
            optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay_l2))
    if args.weight_decay_l1>0:
        optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay_l1))
    # learning rate of pretrained layers
    if args.optimizer in ['Adam','AdaBound','Eve']:
        for func_name in model.predictor.base._children:
            for param in model.predictor.base[func_name].params():
                param.update_rule.hyperparam.alpha *= args.tuning_rate
    else:
        for func_name in model.predictor.base._children:
            for param in model.predictor.base[func_name].params():
                param.update_rule.hyperparam.lr *= args.tuning_rate

    # load model parameters from file
    if args.model:
        print('Load model from: ', args.model)
        chainer.serializers.load_npz(args.model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # data iterators
    train_iter = iterators.MultithreadIterator(train, args.batchsize, shuffle=True, n_threads=args.loaderjob)
    test_iter = iterators.MultithreadIterator(test, args.batchsize, repeat=False, shuffle=False, n_threads=args.loaderjob)
#    train_iter = iterators.MultiprocessIterator(train, args.batchsize, shuffle=True, n_processes=args.loaderjob)
#    test_iter = iterators.MultiprocessIterator(test, args.batchsize, repeat=False, shuffle=False, n_processes=args.loaderjob)

    # set up trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=convert.ConcatWithAsyncTransfer())

    log_interval = 1, 'epoch'
    val_interval = args.vis_freq, 'epoch'
#    val_interval = 10, 'iteration'
    if args.early_stopping:
        stop_trigger = triggers.EarlyStoppingTrigger(
            monitor='validation/main/loss',
            check_trigger=(args.early_stopping, 'epoch'),
            max_trigger=(args.epoch, 'epoch'))
    else:
        stop_trigger = (args.epoch, 'epoch')
    trainer = training.Trainer(updater, stop_trigger, out=args.outdir)

    frequency = args.epoch if args.snapshot == -1 else max(1, args.snapshot)
    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}.npz'), trigger=(frequency, 'epoch'))

    ## learning rate decay
    if args.lr_drop>1:
        if args.optimizer in ['SGD','Momentum','AdaGrad','RMSprop']:
            trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(args.epoch/args.lr_drop, 'epoch'))
        elif args.optimizer in ['Adam','AdaBound','Eve']:
                trainer.extend(extensions.ExponentialShift('alpha', 0.5), trigger=(args.epoch/args.lr_drop, 'epoch'))

    # plots
    trainer.extend(extensions.LogReport(trigger=log_interval))
#    trainer.extend(extensions.dump_graph('main/loss'))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy','myval/Acc'],
                                  'epoch', file_name='accuracy.png'))
        trainer.extend(extensions.PlotReport(['myval/FP','myval/FN'],
                                  'epoch', file_name='val_FPFN.png'))
        trainer.extend(extensions.PlotReport(['myval/score'],
                                  'epoch', file_name='val_score.png'))

    trainer.extend(extensions.PrintReport([
            'epoch', 'main/loss', 'main/accuracy','validation/main/loss',#'validation/main/accuracy',
            'myval/Acc','myval/FN','myval/FP','myval/score'
            # ,'elapsed_time', 'lr'
         ]),trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(Evaluator(test_iter, model, params={'args': args, 'test': test}, device=args.gpu),trigger=val_interval)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),trigger=val_interval)

    if not args.predict:
        trainer.run()

    # final output
    print("predicting: {} entries...".format(len(test)))
    resultfn = "result_{}.csv".format(dtime)
    ac,tp,fn,fp,score = pred(test,args,model,resultfn)
    print("Acc {}, TP {}, FN {}, FP {}, score {}".format(ac,tp,fn,fp,score))

    if not args.predict:
        os.makedirs(dirname, exist_ok=True)
        txts = [os.path.basename(f) for f in glob.glob(os.path.join(args.outdir,"*.txt"))]
        pngs = [os.path.basename(f) for f in glob.glob(os.path.join(args.outdir,"*.png"))]
        for f in [resultfn,"log","args"]+txts+pngs:
            shutil.copyfile(os.path.join(args.outdir,f),os.path.join(dirname,f))

if __name__ == '__main__':
    main()
