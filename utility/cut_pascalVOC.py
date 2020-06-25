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
parser.add_argument('--margin', type=int, default=20, help='margin for cut image')
args = parser.parse_args()

margin = args.margin

for fn in glob.glob(args.datadir+"/**/*.xml", recursive=True):
    fn1, ext = os.path.splitext(fn)
    print("reading {}...".format(fn))
    with open(fn1+".xml") as fd:
        annot = xmltodict.parse(fd.read())['annotation']
    img = Image.open(fn1+".jpg")
    for obj in annot['object']:
        xmin = int(obj['bndbox']['xmin'])
        xmax = int(obj['bndbox']['xmax'])
        ymin = int(obj['bndbox']['ymin'])
        ymax = int(obj['bndbox']['ymax'])
        cut = img.crop((xmin-margin,ymin-margin,xmax+margin,ymax+margin))
        fname = '{}_{}_{:0>4}_{:0>4}.jpg'.format(fn1,obj['name'],ymin,xmin)
        print(fname)
        cut.save(fname)


