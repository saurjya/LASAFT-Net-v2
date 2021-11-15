import random
from abc import ABCMeta
from pathlib import Path
from torch.utils.data import Dataset
from lasaft.utils.fourier import get_trim_length

import musdb
import numpy as np
import soundfile
import torch

from hydra.utils import to_absolute_path


def check_bbcso_valid(bbcso_train):
    if len(bbcso_train) > 0:
        pass
    else:
        raise Exception('Check bbcso json, something is wrong')

