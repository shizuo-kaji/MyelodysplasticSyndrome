#!/usr/bin/env python

import sys
import os
import argparse
import glob
import xmltodict
from PIL import Image
from chainercv.transforms import resize
from chainercv.utils import read_image,write_image

parser = argparse.ArgumentParser()
parser.add_argument('datadir', default='data', help='Directory containing image/xml files')
args = parser.parse_args()

for fn in glob.glob(args.datadir+"/**/*.xml", recursive=True):
    fn1, ext = os.path.splitext(fn)
    with open(fn1+".xml") as fd:
        annot = xmltodict.parse(fd.read())['annotation']
    if type(annot['object']) != list:
        annot['object'] = [annot['object']]
    for obj in annot['object']:
        xmin = int(obj['bndbox']['xmin'])
        xmax = int(obj['bndbox']['xmax'])
        ymin = int(obj['bndbox']['ymin'])
        ymax = int(obj['bndbox']['ymax'])
        fname = '{}_{:0>3}_{:0>3},{}'.format(os.path.basename(fn1),ymin,xmin,obj['name'])
        print(fname)

