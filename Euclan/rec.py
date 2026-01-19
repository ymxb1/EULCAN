import os
import re

import numpy as np
import torch

import cv2
from typing import Tuple, Dict, Any, Optional, List, Union

from torch import Tensor
from torch.nn import functional as F
from itertools import groupby
from pathlib import Path
from line_split import line_split
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy


def read_img(path: Union[str, Path], gray=False) -> np.ndarray:
    """
    :param path: image file path
    :param gray: whether to return a gray image array
    :return:
        * when `gray==True`, return a gray image, with dim [height, width, 1], with values range from 0 to 255
        * when `gray==False`, return a color image, with dim [height, width, 3], with values range from 0 to 255
    """
    if gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Error loading image: {path}')
        return np.expand_dims(img, -1)
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'Error loading image: {path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prepare_img(
        img_fp: Union[str, Path, torch.Tensor, np.ndarray]
) -> np.ndarray:
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


def _prepare_img(
        img_fp: Union[str, Path, torch.Tensor, np.ndarray]
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


def transform_img(img: np.ndarray) -> torch.Tensor:  # 将灰度图的ndarray数组转换为张量
    target_h_w = [32, 521]
    img = img.transpose((2, 0, 1))  # res: [C, H, W]
    new_img = cv2.resize(img.transpose((1, 2, 0)), (target_h_w[1], target_h_w[0]))
    if img.ndim > new_img.ndim:
        new_img = np.expand_dims(new_img, axis=-1)
    img = new_img.transpose((2, 0, 1))  # -> (C, H, W)
    img = torch.from_numpy(img)
    img = img.to(dtype=torch.float32)
    img = img / 255.0
    return img

def is_id_number(id_number):
    if len(id_number) != 18 and len(id_number) != 15:
        # print('身份证号码长度错误')
        return False
    regularExpression = "(^[1-9]\\d{5}(18|19|20)\\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\\d{3}[0-9Xx]$)|" \
                        "(^[1-9]\\d{5}\\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\\d{3}$)"
    # 假设18位身份证号码:41000119910101123X  410001 19910101 123X
    # ^开头
    # [1-9] 第一位1-9中的一个      4
    # \\d{5} 五位数字           10001（前六位省市县地区）
    # (18|19|20)                19（现阶段可能取值范围18xx-20xx年）
    # \\d{2}                    91（年份）
    # ((0[1-9])|(10|11|12))     01（月份）
    # (([0-2][1-9])|10|20|30|31)01（日期）
    # \\d{3} 三位数字            123（第十七位奇数代表男，偶数代表女）
    # [0-9Xx] 0123456789Xx其中的一个 X（第十八位为校验值）
    # $结尾

    # 假设15位身份证号码:410001910101123  410001 910101 123
    # ^开头
    # [1-9] 第一位1-9中的一个      4
    # \\d{5} 五位数字           10001（前六位省市县地区）
    # \\d{2}                    91（年份）
    # ((0[1-9])|(10|11|12))     01（月份）
    # (([0-2][1-9])|10|20|30|31)01（日期）
    # \\d{3} 三位数字            123（第十五位奇数代表男，偶数代表女），15位身份证不含X
    # $结尾
    if re.match(regularExpression, id_number):
        if len(id_number) == 18:
            n = id_number.upper()
            # 前十七位加权因子
            var = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
            # 这是除以11后，可能产生的11位余数对应的验证码
            var_id = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

            sum = 0
            for i in range(0, 17):
                sum += int(n[i]) * var[i]
            sum %= 11
            if (var_id[sum]) != str(n[17]):
                print("身份证号规则核验失败，校验码应为", var_id[sum], "，当前校验码是：", n[17])
                return
        return id_number
    else:
        return


def ocr_for_single_lines(
        img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
        batch_size: int = 1,
) -> List[Tensor]:
    if len(img_list) == 0:
        return []

    img_list = [prepare_img(img) for img in img_list]
    outs = recognize(img_list, batch_size=batch_size)
    return outs
    # results = []
    # for text, score in outs:
    #     _out = OcrResult(text=text, score=score)
    #     results.append(_out.to_dict())
    # return results


def ocr(
        img_fp: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
        rec_batch_size=1,
        **det_kwargs,
) -> List[Dict[str, Any]]:
    if isinstance(img_fp, Image.Image):  # Image to np.ndarray
        img_fp = np.asarray(img_fp.convert('RGB'))

    img = prepare_img(img_fp)
    if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
        img = 255 - img
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=-1)
    line_imgs = line_split(img, blank=True)
    line_img_list = [line_img for line_img, _ in line_imgs]
    # line_chars_list = ocr_for_single_lines(
    #     line_img_list, batch_size=rec_batch_size
    # )

    return line_img_list


def recognize(
        img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
        batch_size: int = 1,
) -> List[Tensor]:
    if len(img_list) == 0:
        return []

    img_list = [_prepare_img(img) for img in img_list]
    img_list = [transform_img(img) for img in img_list]
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
    #
    # idx = 0
    # sorted_out = []
    # while idx * batch_size < len(sorted_img_list):
    #     imgs = sorted_img_list[idx * batch_size: (idx + 1) * batch_size]

    return sorted_img_list
    #     try:
    #         batch_out = _predict(img_list=imgs)
    #     except Exception as e:
    #         # 对于太小的图片，如宽度小于8，会报错
    #         batch_out = {'preds': [([''], 0.0)] * len(imgs)}
    #     sorted_out.extend(batch_out['preds'])
    #     idx += 1
    # out = [None] * len(sorted_out)
    # for idx, pred in zip(sorted_idx_list, sorted_out):
    #     out[idx] = pred
    #
    # res = []
    # for line in out:
    #     chars, prob = line
    #     chars = [c if c != '<space>' else ' ' for c in chars]
    #     res.append((''.join(chars), prob))

    # return res



def pad_img_seq(img_list: List[torch.Tensor], padding_value=0) -> torch.Tensor:
    img_list = [img.permute((2, 0, 1)) for img in img_list]  # [W, C, H]
    imgs = pad_sequence(
        img_list, batch_first=True, padding_value=padding_value
    )  # [B, W_max, C, H]
    return imgs.permute((0, 2, 3, 1))  # [B, C, H, W_max]


def mask_by_candidates(
        logits: torch.Tensor,
        candidates: Optional[Union[str, List[str]]],
        vocab: List[str],
        letter2id: Dict[str, int],
):
    if candidates is None:
        return logits

    _candidates = [letter2id[word] for word in candidates]
    _candidates.sort()
    _candidates = torch.tensor(_candidates, dtype=torch.int64)

    candidates = torch.zeros(
        (len(vocab) + 1,), dtype=torch.bool, device=logits.device
    )
    candidates[_candidates] = True
    candidates[-1] = True  # 间隔符号/填充符号，必须为真
    candidates = candidates.unsqueeze(0).unsqueeze(0)  # 1 x 1 x (vocab_size+1)
    logits.masked_fill_(~candidates, -100.0)
    return logits


def load_vocab(vocab_fp: str) -> List[str]:
    with open(vocab_fp, 'r', encoding='utf-8') as file:
        vocab = [line.strip() for line in file]
    return vocab


def gen_length_mask(lengths: torch.Tensor, mask_size: Union[Tuple, Any]):
    """ see how it is used """
    labels = torch.arange(mask_size[-1], device=lengths.device, dtype=torch.long)
    while True:
        if len(labels.shape) >= len(mask_size):
            break
        labels = labels.unsqueeze(0)
        lengths = lengths.unsqueeze(-1)
    mask = labels < lengths
    return ~mask


def ctc_best_path(
        logits: torch.Tensor,
        vocab: List[str],
        input_lengths: Optional[torch.Tensor] = None,
        blank: int = 0,
) -> List[Tuple[List[str], float]]:
    # compute softmax
    probs = F.softmax(logits.permute(0, 2, 1), dim=1)
    # get char indices along best path
    best_path = torch.argmax(probs, dim=1)  # [N, T]

    if input_lengths is not None:
        length_mask = gen_length_mask(input_lengths, probs.shape).to(
            device=probs.device
        )  # [N, 1, T]
        probs.masked_fill_(length_mask, 1.0)
        best_path.masked_fill_(length_mask.squeeze(1), blank)

    # define word proba as min proba of sequence
    probs, _ = torch.max(probs, dim=1)  # [N, T]
    probs, _ = torch.min(probs, dim=1)  # [N]

    words = []
    for sequence in best_path:
        # collapse best path (using itertools.groupby), map to chars, join char list to string
        collapsed = [vocab[k] for k, _ in groupby(sequence) if k != blank]
        words.append(collapsed)

    return list(zip(words, probs.tolist()))


class OcrResult:
    def __init__(self, text: str, score: float, position: Optional[np.ndarray] = None,
                 cropped_img: Optional[np.ndarray] = None):
        self.text = text
        self.score = score
        self.position = position
        self.cropped_img = cropped_img

    def to_dict(self):
        res = deepcopy(self.__dict__)
        if self.position is None:
            res.pop('position')
        if self.cropped_img is None:
            res.pop('cropped_img')
        return res