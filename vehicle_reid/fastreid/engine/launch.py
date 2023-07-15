# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on:
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py


import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
