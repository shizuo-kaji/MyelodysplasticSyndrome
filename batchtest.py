# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function
import argparse
import subprocess
import sys


argsv=" ".join(sys.argv[1:])

## windows size
for i in range(0,5):
    cmd = "python finetune_mds.py -t cv{} -bg 00background.jpg {}".format(i,argsv)
#    cmd = "python finetune_mds.py -t cv{} -bg 00background.jpg {}".format(i,argsv)
    print(cmd)
    subprocess.call(cmd, shell=True)
