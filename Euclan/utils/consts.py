# coding: utf-8

import os
import string
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Set, Dict, Any, Optional, Union
import logging
from copy import deepcopy

from .__version__ import __version__

logger = logging.getLogger(__name__)


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '2.2.*'，对应的 MODEL_VERSION 都是 '2.2'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
DOWNLOAD_SOURCE = os.environ.get('CNOCR_DOWNLOAD_SOURCE', 'CN')

IMG_STANDARD_HEIGHT = 32
CN_VOCAB_FP = Path(__file__).parent.absolute() / 'label_cn.txt'
NUMBER_VOCAB_FP = Path(__file__).parent.absolute() / 'label_number.txt'

ENCODER_CONFIGS = {
    'EULCAN': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 4*128 = 512
        'growth_rate': 32,
        'block_config': [6, 12, 24, 16],
        'num_init_features': 64,
        'out_length': 4096,  # 输出的向量长度为 4*128 = 512
    },
    'EULCAN_lite_136': {  # 长度压缩至 1/8（seq_len == 35）; #params, with fc: 680 K, with gru: 1.4 M
        'growth_rate': 32,
        'block_config': [1, 3, 6],
        'num_init_features': 64,
        # 'out_length': 256,
        'out_length': 528,
    },
}

DECODER_CONFIGS = {
    'fc': {'hidden_size': 128, 'dropout': 0.1},
}

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation
