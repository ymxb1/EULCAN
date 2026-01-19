# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 数据
cer = [
    [0.1900627166, 0.108146064],
    [0.1480945498, 0.06741572917],
    [0.1239749119, 0.04073033854]
]
complete_match = [
    [0.5974842749, 0.7415730295],
    [0.6540880483, 0.7977528045],
    [0.7075471676, 0.8764044895]
]
models = ['densenet_lite_136-fc', 'densenet_lite_136-gru', 'densenet_lite_666-gru_large']
datasets = ['real-number-index/dev.tsv', 'ocr-2023-08-19/dev.tsv']
barWidth = 0.25

# 颜色和模型对应关系
colors_new = {
    'densenet_lite_136-fc': '#247266',
    'densenet_lite_136-gru': '#F5450C',
    'densenet_lite_666-gru_large': '#D200DB'
}

# 将数据转换为百分比
cer_percent = np.array(cer) * 100
complete_match_percent = np.array(complete_match) * 100

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 画 CER 柱状图
r1 = np.arange(len(datasets))
for i, model in enumerate(models):
    axes[0].bar(r1, [cer_percent[i][0], cer_percent[i][1]], color=colors_new[model], width=barWidth, edgecolor='grey', label=model)
    r1 = [x + barWidth for x in r1]

# 添加标签
axes[0].set_xlabel('Datasets', fontweight='bold', fontsize=15)
axes[0].set_xticks([r + barWidth for r in range(len(datasets))])
axes[0].set_xticklabels(datasets)
axes[0].set_title('Char Error Rate ↓')  # 添加向下箭头
axes[0].legend()
axes[0].set_yticklabels(['{:.1f}%'.format(x) for x in axes[0].get_yticks()])

# 重置位置
r1 = np.arange(len(datasets))

# 画 Complete Match 柱状图
for i, model in enumerate(models):
    axes[1].bar(r1, [complete_match_percent[i][0], complete_match_percent[i][1]], color=colors_new[model], width=barWidth, edgecolor='grey', label=model)
    r1 = [x + barWidth for x in r1]

# 添加标签
axes[1].set_xlabel('Datasets', fontweight='bold', fontsize=15)
axes[1].set_xticks([r + barWidth for r in range(len(datasets))])
axes[1].set_xticklabels(datasets)
axes[1].set_title('Complete Match ↑')  # 添加向上箭头
axes[1].set_ylim(50, 100)
axes[1].legend()
axes[1].set_yticklabels(['{:.1f}%'.format(x) for x in axes[1].get_yticks()])

plt.tight_layout()
plt.show()

