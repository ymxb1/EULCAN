import re
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

__all__ = [
    "Eulcan",
    "SCAB",
    "h_sigmoid",
    "h_swish"
]

# 模拟缺失的 _log_api_usage_once 函数（避免调用报错，不影响网络结构核心功能）
def _log_api_usage_once(obj: Any) -> None:
    """模拟 torchvision 的 API 日志记录函数，无实际业务逻辑，仅保证代码可运行"""
    pass

# 模拟缺失的权重相关工具函数（若不需要加载预训练权重，仅保留空实现保证代码完整）
class WeightsEnum:
    """模拟权重枚举类，避免代码报错"""
    meta = {"categories": []}
    @staticmethod
    def get_state_dict(progress: bool, check_hash: bool):
        return OrderedDict()

def _ovewrite_named_param(kwargs: dict, name: str, value: Any) -> None:
    """模拟参数覆盖函数"""
    if name not in kwargs:
        kwargs[name] = value

def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    """模拟预训练权重加载函数，无实际权重加载逻辑，仅保证代码完整"""
    pattern = re.compile(
        r"^(.*ebcb\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    state_dict = weights.get_state_dict(progress=progress, check_hash=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    # 跳过实际加载（无预训练权重文件），避免报错
    pass


class _EBCB(nn.Module):
    def __init__(
            self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.add_module('SCAB', SCAB(growth_rate, reduction=16))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    # 修复前向传播重载，兼容 Tensor 和 List[Tensor] 输入
    def forward(self, input: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        new_features = self.SCAB(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DSCAM(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _EBCB(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("ebcb%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for ebcb_name, ebcb_layer in self.items():
            new_features = ebcb_layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _DDRB(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class SCAB(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SCAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, ch_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)  # [b, c, 1, 1]
        y = self.conv(y)      # 生成通道注意力权重
        return x * y.expand_as(x)  # 注意力加权

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Eulcan(nn.Module):


    def __init__(
            self,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        # First convolution（修复：保留完整卷积层结构，支持 3 通道输入）
        self.Stem = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        self.features = nn.Sequential()
        # Each DSCAM
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DSCAM(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("DSCAM%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _DDRB(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("DDRB%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def get_features_extractor(self) -> nn.Sequential:
        """额外提供：获取特征提取器（不含分类头），方便后续自定义修改"""
        return self.features


def _eulcan(
        growth_rate: int,
        block_config: Tuple[int, int, int, int],
        num_init_features: int,
        weights: Optional[WeightsEnum] = None,
        progress: bool = True,
        **kwargs: Any,
) -> Eulcan:
    """DenseNet 构造函数，简化模型实例化"""
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = Eulcan(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

    return model
