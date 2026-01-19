# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, List, Any, Dict, Optional, Collection, Tuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from cnstd.consts import AVAILABLE_MODELS as DET_AVAILABLE_MODELS
from cnstd import CnStd
from cnstd.utils import data_dir as det_data_dir

from utils.utils import data_dir, read_img
from line_split import line_split
from recognizer import Recognizer

logger = logging.getLogger(__name__)

DET_MODLE_NAMES, _ = zip(*DET_AVAILABLE_MODELS.all_models())
DET_MODLE_NAMES = set(DET_MODLE_NAMES)

# 新增：默认词表路径（从原 consts.py 中提取）
IMG_STANDARD_HEIGHT = 32
IMG_STANDARD_WIDTH = 521
CN_VOCAB_FP = Path(__file__).parent.absolute() / 'label_cn.txt'
NUMBER_VOCAB_FP = Path(__file__).parent.absolute() / 'label_number.txt'


@dataclass
class OcrResult(object):
    text: str
    score: float
    position: Optional[np.ndarray] = None
    cropped_img: np.ndarray = None

    def to_dict(self):
        res = deepcopy(self.__dict__)
        if self.position is None:
            res.pop('position')
        if self.cropped_img is None:
            res.pop('cropped_img')
        return res


class CnOcr(object):
    def __init__(
        self,
        rec_model_name: str = 'densenet_lite_136-gru',
        *,
        det_model_name: str = 'ch_PP-OCRv3_det',
        cand_alphabet: Optional[Union[Collection, str]] = None,
        context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
        rec_model_fp: Optional[str] = None,
        rec_model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        rec_vocab_fp: Optional[Union[str, Path]] = None,
        rec_more_configs: Optional[Dict[str, Any]] = None,
        rec_root: Union[str, Path] = data_dir(),
        det_model_fp: Optional[str] = None,
        det_model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        det_more_configs: Optional[Dict[str, Any]] = None,
        det_root: Union[str, Path] = det_data_dir(),
        **kwargs: object,
    ) -> object:

        if kwargs.get('model_name') is not None and rec_model_name is None:
            # 兼容前面的版本
            rec_model_name = kwargs.get('model_name')

        rec_cls = Recognizer

        if rec_vocab_fp is None:
            # 根据模型名称自动匹配词表（可选逻辑）
            if 'number' in rec_model_name:
                rec_vocab_fp = NUMBER_VOCAB_FP
            else:
                rec_vocab_fp = CN_VOCAB_FP

        rec_more_configs = rec_more_configs or dict()
        self.rec_model = rec_cls(
            model_name=rec_model_name,
            model_backend=rec_model_backend,
            cand_alphabet=cand_alphabet,
            context=context,
            model_fp=rec_model_fp,
            root=rec_root,
            vocab_fp=rec_vocab_fp,
            **rec_more_configs,
        )

        self.det_model = None
        if det_model_name in DET_MODLE_NAMES:
            det_more_configs = det_more_configs or dict()
            self.det_model = CnStd(
                det_model_name,
                model_backend=det_model_backend,
                context=context,
                model_fp=det_model_fp,
                root=det_root,
                **det_more_configs,
            )

    def ocr(
        self,
        img_fp: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
        rec_batch_size=1,
        return_cropped_image=True,
        **det_kwargs,
    ) -> List[Dict[str, Any]]:

        if isinstance(img_fp, Image.Image):  # Image to np.ndarray
            img_fp = np.asarray(img_fp.convert('RGB'))

        if self.det_model is not None:
            return self._ocr_with_det_model(
                img_fp, rec_batch_size, return_cropped_image, **det_kwargs
            )

        img = self._prepare_img(img_fp)

        if min(img.shape[0], img.shape[1]) < 2:
            return []
        if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
            img = 255 - img
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=-1)
        line_imgs = line_split(img, blank=True)
        line_img_list = [line_img for line_img, _ in line_imgs]
        line_chars_list = self.ocr_for_single_lines(
            line_img_list, batch_size=rec_batch_size
        )
        if return_cropped_image:
            for _out, line_img in zip(line_chars_list, line_img_list):
                _out['cropped_img'] = line_img

        return line_chars_list

    def _ocr_with_det_model(
        self,
        img: Union[str, Path, torch.Tensor, np.ndarray],
        rec_batch_size: int,
        return_cropped_image: bool,
        **det_kwargs,
    ) -> List[Dict[str, Any]]:
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 1:
                # (H, W, 1) -> (H, W)
                img = img.squeeze(-1)
            if len(img.shape) == 2:
                # (H, W) -> (H, W, 3)
                img = np.array(Image.fromarray(img).convert('RGB'))

        box_infos = self.det_model.detect(img, **det_kwargs)

        cropped_img_list = [
            box_info['cropped_img'] for box_info in box_infos['detected_texts']
        ]
        ocr_outs = self.ocr_for_single_lines(
            cropped_img_list, batch_size=rec_batch_size
        )
        results = []
        for box_info, ocr_out in zip(box_infos['detected_texts'], ocr_outs):
            _out = OcrResult(**ocr_out)
            _out.position = box_info['box']
            if return_cropped_image:
                _out.cropped_img = box_info['cropped_img']
            results.append(_out.to_dict())

        return results

    def _prepare_img(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        预处理图片为统一格式
        """
        img = img_fp
        if isinstance(img_fp, (str, Path)):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp, gray=False)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        elif len(img.shape) == 3:
            assert img.shape[2] in (1, 3)

        if img.dtype != np.dtype('uint8'):
            img = img.astype('uint8')
        return img

    def ocr_for_single_line(
            self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> Tuple[Tuple[List[str], float], dict]:
        """
        Recognize characters from an image with only one-line characters.
        """
        img = self._prepare_img(img_fp)
        res, time_stats = self.rec_model.recognize([img])  # 调用改造后的recognize方法
        return (res[0] if res else ([''], 0.0)), time_stats

    def ocr_for_single_lines(
            self,
            img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
            batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Batch recognize characters from a list of one-line-characters images.
        """
        if len(img_list) == 0:
            return []

        img_list = [self._prepare_img(img) for img in img_list]
        # 接收 recognize() 返回的 (outs, time_stats) 二元组
        outs, time_stats = self.rec_model.recognize(img_list, batch_size=batch_size)

        results = []
        for text, score in outs:
            _out = OcrResult(text=text, score=score)
            results.append(_out.to_dict())

        return results
