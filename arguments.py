import numpy as np
import chainer.functions as F
from consts import dtypes, optim
import os
from datetime import datetime as dt
from chainerui.utils import save_args
import argparse

def arguments():
    parser = argparse.ArgumentParser(description='MDS classifier/regressor')
    parser.add_argument('--train', '-t', default="cv0",help='Path to csv file')
    parser.add_argument('--case', '-c', default="all", help='case')
    parser.add_argument('--target', '-tg', default=-1, type=int, help='target label')
    parser.add_argument('--target_id', '-tid', default=13, type=int, help='target id')  # 'L'
    parser.add_argument('--root', '-R', default="image", help='Path to image files')
    parser.add_argument('--val', help='Path to validation csv file')
    parser.add_argument('--arch', '-a', default="resnet152", help='DNN architecture')
    parser.add_argument('--global_pool', '-gp', default="none", help='global pooling')
    parser.add_argument('--augment', '-ag', type=str, default="equal", help='data augmentation scheme')
    parser.add_argument('--scaling_jitter', '-sj', type=float, default=0.2,help='scaling between [1.0-x,1.0+x]')
    parser.add_argument('--ensemble', '-en', type=str, default="16", help='number of rotated images for ensemble prediction')
    parser.add_argument('--regression', '-rg', default=True, help='regression/classification')
    parser.add_argument('--bg_image', '-bg', default=None, help='background image')
    parser.add_argument('--local_average_subtraction', '-las', type=int, default=0, help='preprocess images with local average subtraction with this kernel size')
    parser.add_argument('--save_preprocessed_image', '-sp', default=None,  help='save preprocessed images under this dir')

    parser.add_argument('--early_stopping', '-es', type=int, default=0, help='')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,help='Learning rate')
    parser.add_argument('--tuning_rate', '-tr', type=float, default=0.1,help='learning rate for pretrained layers')
    parser.add_argument('--lr_drop', '-lrd', type=int, default=0,
                        help='strategy for learning rate decay')

    parser.add_argument('--batchsize', '-b', type=int, default=10,help='Number of samples in each mini-batch')
    parser.add_argument('--vis_freq', '-vf', type=int, default=3,help='frequency of evaluation in epochs')
    parser.add_argument('--layer', '-l', type=str, choices=['res5','pool5'], default='pool5',help='output layer of the pretrained model')

    parser.add_argument('--loss', type=str, default='rmse',help='loss')
    parser.add_argument('--class_weight', '-cw', type=float, default=10,help='rmse weight for the target label')
    parser.add_argument('--loss_exponent', '-le', type=float, default=4,help='exponent for the loss')

    parser.add_argument('--fch', '-fch', type=int, nargs="*", default=[],
                        help='numbers of channels for the last fc layers')
    parser.add_argument('--epoch', '-e', type=int, default=120,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--snapshot', '-s', type=int, default=-1,
                        help='snapshot interval')
    parser.add_argument('--scale_to', '-sc', type=int, default=232,
                        help='images are scale to')
    parser.add_argument('--model', '-m',
                        help='Initialize the model from given file')
    parser.add_argument('--random', '-rt', type=int, default=8,
                        help='random translation')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', type=int, default=5,
                        help='Number of parallel data loading processes')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')
    parser.add_argument('--predict', '-p', action='store_true', help='prediction with a specified model')
    parser.add_argument('--dropout', '-dr', type=float, default=0.5,
                        help='dropout ratio for the FC layers')
    parser.add_argument('--weight_decay_l1', '-wd1', type=float, default=0,
                        help='weight decay for regularization')
    parser.add_argument('--weight_decay_l2', '-wd2', type=float, default=1e-6,
                        help='weight decay for regularization')
    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    args = parser.parse_args()

    return args
