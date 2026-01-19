# coding: utf-8


import os
import logging
from glob import glob
from typing import Union, List, Tuple, Optional, Collection
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from cnstd.utils import get_model_file

from utils.consts import MODEL_VERSION, DOWNLOAD_SOURCE
from models.ocr_model import OcrModel
from utils.utils import (
    data_dir,
    read_charset,
    check_model_name,
    check_context,
    read_img,
    load_model_params,
    resize_img,
    pad_img_seq,
    to_numpy,
    get_default_ort_providers,
)
from utils.aug import NormalizeAug
from models.ctc import CTCPostProcessor

logger = logging.getLogger(__name__)

# 自定义默认模型配置（替代AVAILABLE_MODELS）
DEFAULT_MODEL_CONFIGS = {
    'densenet_lite_136-gru': {
        'pytorch': {'epoch': 100, 'vocab_fp': 'label_cn.txt', 'url': ''},
        'onnx': {'epoch': 100, 'vocab_fp': 'label_cn.txt', 'url': ''}
    }
}


def gen_model(model_name, vocab):
    check_model_name(model_name)
    model = OcrModel.from_name(model_name, vocab)
    return model


class Recognizer(object):
    MODEL_FILE_PREFIX = 'cnocr-v{}'.format(MODEL_VERSION)

    def __init__(
            self,
            model_name: str = 'densenet_lite_136-gru',
            *,
            cand_alphabet: Optional[Union[Collection, str]] = None,
            context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
            model_fp: Optional[str] = None,
            model_backend: str = 'onnx',  # ['pytorch', 'onnx']
            root: Union[str, Path] = data_dir(),
            vocab_fp: Optional[Union[str, Path]] = None,
            **kwargs,
    ):

        model_backend = model_backend.lower()
        assert model_backend in ('pytorch', 'onnx')
        if 'name' in kwargs:
            logger.warning(
                'param `name` is useless and deprecated since version %s'
                % MODEL_VERSION
            )
        check_model_name(model_name)
        check_context(context)

        self._model_name = model_name
        self._model_backend = model_backend
        if context == 'gpu':
            context = 'cuda'
        self.context = context

        try:
            self._assert_and_prepare_model_files(model_fp, root)
        except NotImplementedError:
            logger.warning(
                'no available model is found for name %s and backend %s'
                % (self._model_name, self._model_backend)
            )
            self._model_backend = (
                'onnx' if self._model_backend == 'pytorch' else 'pytorch'
            )
            logger.warning(
                'trying to use name %s and backend %s'
                % (self._model_name, self._model_backend)
            )
            self._assert_and_prepare_model_files(model_fp, root)

        if vocab_fp is None:
            vocab_fp = self._get_default_vocab_fp(self._model_name, self._model_backend)
        self._vocab, self._letter2id = read_charset(vocab_fp)
        self.postprocessor = CTCPostProcessor(vocab=self._vocab)

        self._candidates = None
        self.set_cand_alphabet(cand_alphabet)

        self._model = self._get_model(
            context, ort_providers=kwargs.get('ort_providers')
        )

    def _get_default_vocab_fp(self, model_name, model_backend):
        """获取默认词表文件路径（替代AVAILABLE_MODELS.get_vocab_fp）"""
        if model_name not in DEFAULT_MODEL_CONFIGS:
            raise ValueError(f"Unsupported model name: {model_name}")
        if model_backend not in DEFAULT_MODEL_CONFIGS[model_name]:
            raise ValueError(f"Unsupported backend {model_backend} for model {model_name}")

        # 拼接默认词表路径（需根据实际项目结构调整）
        vocab_name = DEFAULT_MODEL_CONFIGS[model_name][model_backend]['vocab_fp']
        vocab_fp = os.path.join(data_dir(), MODEL_VERSION, model_name, vocab_name)

        # 如果默认路径不存在，使用项目内置的默认词表
        if not os.path.exists(vocab_fp):
            vocab_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocabs', vocab_name)

        if not os.path.exists(vocab_fp):
            raise FileNotFoundError(f"Default vocab file not found for model {model_name}, backend {model_backend}")

        return vocab_fp

    def _get_model_epoch(self, model_name, model_backend):
        """获取模型训练轮数（替代AVAILABLE_MODELS.get_epoch）"""
        if model_name not in DEFAULT_MODEL_CONFIGS:
            return None
        if model_backend not in DEFAULT_MODEL_CONFIGS[model_name]:
            return None
        return DEFAULT_MODEL_CONFIGS[model_name][model_backend]['epoch']

    def _get_model_url(self, model_name, model_backend):
        """获取模型下载地址（替代AVAILABLE_MODELS.get_url）"""
        if model_name not in DEFAULT_MODEL_CONFIGS:
            raise NotImplementedError(f"Model {model_name} is not supported")
        if model_backend not in DEFAULT_MODEL_CONFIGS[model_name]:
            raise NotImplementedError(f"Backend {model_backend} for model {model_name} is not supported")

        url = DEFAULT_MODEL_CONFIGS[model_name][model_backend]['url']
        if not url:
            raise NotImplementedError(f"No download url for model {model_name}, backend {model_backend}")
        return url

    def _assert_and_prepare_model_files(self, model_fp, root):
        self._model_file_prefix = '{}-{}'.format(
            self.MODEL_FILE_PREFIX, self._model_name
        )
        model_epoch = self._get_model_epoch(self._model_name, self._model_backend)

        if model_epoch is not None:
            self._model_file_prefix = '%s-epoch=%03d' % (
                self._model_file_prefix,
                model_epoch,
            )

        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, self._model_name)
        model_ext = 'ckpt' if self._model_backend == 'pytorch' else 'onnx'
        fps = glob('%s/%s*.%s' % (self._model_dir, self._model_file_prefix, model_ext))
        if len(fps) > 1:
            raise ValueError(
                'multiple %s files are found in %s, not sure which one should be used'
                % (model_ext, self._model_dir)
            )
        elif len(fps) < 1:
            logger.warning('no %s file is found in %s' % (model_ext, self._model_dir))
            # 检查模型是否支持下载（替代AVAILABLE_MODELS的成员检查）
            try:
                self._get_model_url(self._model_name, self._model_backend)
            except NotImplementedError:
                raise NotImplementedError(
                    '%s is not a downloadable model'
                    % ((self._model_name, self._model_backend),)
                )
            url = self._get_model_url(self._model_name, self._model_backend)
            get_model_file(
                url, self._model_dir, download_source=DOWNLOAD_SOURCE
            )  # download the .zip file and unzip
            fps = glob(
                '%s/%s*.%s' % (self._model_dir, self._model_file_prefix, model_ext)
            )

        self._model_fp = fps[0]

    def _get_model(self, context, ort_providers=None):
        logger.info('use model: %s' % self._model_fp)
        if self._model_backend == 'pytorch':
            model = gen_model(self._model_name, self._vocab)
            model.eval()
            model.to(self.context)
            model = load_model_params(model, self._model_fp, context)
        elif self._model_backend == 'onnx':
            import onnxruntime as ort

            if ort_providers is None:
                ort_providers = get_default_ort_providers()
            logger.debug(f'ort providers: {ort_providers}')
            model = ort.InferenceSession(self._model_fp, providers=ort_providers)
        else:
            raise NotImplementedError(f'{self._model_backend} is not supported yet')

        return model

    def set_cand_alphabet(self, cand_alphabet: Optional[Union[Collection, str]]):
        """
        设置待识别字符的候选集合。

        Args:
            cand_alphabet (Optional[Union[Collection, str]]): 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围

        Returns:
            None

        """
        if cand_alphabet is None:
            self._candidates = None
        else:
            cand_alphabet = [
                word if word != ' ' else '<space>' for word in cand_alphabet
            ]
            excluded = set(
                [word for word in cand_alphabet if word not in self._letter2id]
            )
            if excluded:
                logger.warning(
                    'chars in candidates are not in the vocab, ignoring them: %s'
                    % excluded
                )
            candidates = [word for word in cand_alphabet if word in self._letter2id]
            self._candidates = None if len(candidates) == 0 else candidates
            logger.debug('candidate chars: %s' % self._candidates)

    def _prepare_img(
            self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> np.ndarray:

        img = img_fp
        if isinstance(img_fp, (str, Path)):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                # color to gray
                img = np.expand_dims(np.array(Image.fromarray(img).convert('L')), -1)
            elif img.shape[2] != 1:
                raise ValueError(
                    'only images with shape [height, width, 1] (gray images), '
                    'or [height, width, 3] (RGB-formated color images) are supported'
                )

        if img.dtype != np.dtype('uint8'):
            img = img.astype('uint8')
        return img

    def ocr_for_single_line(
            self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> Tuple[List[str], float]:
        """
        Recognize characters from an image with only one-line characters.

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]):
                image file path; or image torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (color image).

        Returns:
            tuple: (list of chars, prob), such as (['你', '好'], 0.80)
        """
        img = self._prepare_img(img_fp)
        res = self.ocr_for_single_lines([img])
        return res[0]

    def recognize(
            self,
            img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
            batch_size: int = 1,
    ) -> Tuple[List[Tuple[str, float]], dict]:  # 修改返回值：增加耗时统计字典
        """
        Batch recognize characters from a list of one-line-characters images.

        Args:
            img_list (List[Union[str, Path, torch.Tensor, np.ndarray]]):
                list of images, in which each element should be a line image array,
                with type torch.Tensor or np.ndarray.
                Each element should be a tensor with values ranging from 0 to 255,
                and with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (RGB-format color image).
                注：img_list 不宜包含太多图片，否则同时导入这些图片会消耗很多内存。
            batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。

        Returns:
            tuple:
                - list: list of (chars, prob), such as [('第一行', 0.80), ('第二行', 0.75)]
                - dict: 各阶段耗时统计，包含 preprocess_time、inference_time、postprocess_time、total_time
        """
        # 初始化耗时统计字典
        time_stats = {
            'preprocess_time': 0.0,
            'inference_time': 0.0,
            'postprocess_time': 0.0,
            'total_time': 0.0
        }
        total_start = time.time()  # 记录整体开始时间

        if len(img_list) == 0:
            time_stats['total_time'] = time.time() - total_start
            return [], time_stats

        # ---------------------- 阶段1：图片预处理（计时）----------------------
        preprocess_start = time.time()
        img_list = [self._prepare_img(img) for img in img_list]
        img_list = [self._transform_img(img) for img in img_list]

        should_sort = batch_size > 1 and len(img_list) // batch_size > 1

        if should_sort:
            # 把图片按宽度从小到大排列，提升效率
            sorted_idx_list = sorted(
                range(len(img_list)), key=lambda i: img_list[i].shape[2]
            )
            sorted_img_list = [img_list[i] for i in sorted_idx_list]
        else:
            sorted_idx_list = range(len(img_list))
            sorted_img_list = img_list
        preprocess_end = time.time()
        time_stats['preprocess_time'] = preprocess_end - preprocess_start

        # ---------------------- 阶段2：模型批量推理（计时）----------------------
        inference_start = time.time()
        idx = 0
        sorted_out = []
        while idx * batch_size < len(sorted_img_list):
            imgs = sorted_img_list[idx * batch_size: (idx + 1) * batch_size]
            try:
                batch_out = self._predict(imgs)
            except Exception as e:
                # 对于太小的图片，如宽度小于8，会报错
                batch_out = {'preds': [([''], 0.0)] * len(imgs)}
            sorted_out.extend(batch_out['preds'])
            idx += 1
        out = [None] * len(sorted_out)
        for idx, pred in zip(sorted_idx_list, sorted_out):
            out[idx] = pred
        inference_end = time.time()
        time_stats['inference_time'] = inference_end - inference_start

        # ---------------------- 阶段3：结果后处理（计时）----------------------
        postprocess_start = time.time()
        res = []
        for line in out:
            chars, prob = line
            chars = [c if c != '<space>' else ' ' for c in chars]
            res.append((''.join(chars), prob))
        postprocess_end = time.time()
        time_stats['postprocess_time'] = postprocess_end - postprocess_start

        # ---------------------- 计算总耗时 & 打印耗时信息 ----------------------
        time_stats['total_time'] = time.time() - total_start
        # 控制台打印精细耗时（DEBUG级别，不干扰原有日志）
        logger.debug(
            f"Recognizer.recognize 精细耗时统计："
            f"预处理 {time_stats['preprocess_time']:.6f}s | "
            f"推理 {time_stats['inference_time']:.6f}s | "
            f"后处理 {time_stats['postprocess_time']:.6f}s | "
            f"总耗时 {time_stats['total_time']:.6f}s"
        )
        # 额外打印普通信息（控制台可见，无需配置日志级别）
        print(
            f"[Recognizer 内部耗时] 预处理：{time_stats['preprocess_time']:.4f}s | 推理：{time_stats['inference_time']:.4f}s | 后处理：{time_stats['postprocess_time']:.4f}s")

        return res, time_stats

    def _transform_img(self, img: np.ndarray) -> torch.Tensor:
        """
        Args:
            img: image array with type torch.Tensor or np.ndarray,
            with shape [height, width] or [height, width, channel].
            channel shoule be 1 (gray image) or 3 (color image).

        Returns:
            torch.Tensor: with shape (1, height, width)
        """
        img = resize_img(img.transpose((2, 0, 1)))  # res: [C, H, W]
        return NormalizeAug()(img).to(device=torch.device(self.context))

    def _predict(self, img_list: List[torch.Tensor]):
        """单次批量图片推理，返回模型输出结果"""
        predict_start = time.time()  # 新增：推理开始时间
        img_lengths = torch.tensor([img.shape[2] for img in img_list])
        imgs = pad_img_seq(img_list)

        if self._model_backend == 'pytorch':
            with torch.no_grad():
                out = self._model(
                    imgs, img_lengths, candidates=self._candidates, return_preds=True
                )
        else:  # onnx
            out = self._onnx_predict(imgs, img_lengths)

        # 新增：打印单次批量推理耗时
        predict_time = time.time() - predict_start
        logger.debug(f"单次批量推理耗时（{len(img_list)} 张图片）：{predict_time:.6f}s")
        return out

    def _onnx_predict(self, imgs, img_lengths):
        ort_session = self._model
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(imgs),
            ort_session.get_inputs()[1].name: to_numpy(img_lengths),
        }
        ort_outs = ort_session.run(None, ort_inputs)
        out = {
            'logits': torch.from_numpy(ort_outs[0]),
            'output_lengths': torch.from_numpy(ort_outs[1]),
        }
        out['logits'] = OcrModel.mask_by_candidates(
            out['logits'], self._candidates, self._vocab, self._letter2id
        )

        out["preds"] = self.postprocessor(out['logits'], out['output_lengths'])
        return out
