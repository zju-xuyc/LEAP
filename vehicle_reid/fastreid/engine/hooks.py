# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import datetime
import itertools
import logging
import os
import tempfile
import time
from collections import Counter

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

