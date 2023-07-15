# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import time
import weakref
from typing import Dict

import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel