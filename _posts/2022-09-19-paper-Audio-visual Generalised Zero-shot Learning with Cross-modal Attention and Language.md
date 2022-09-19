---
layout: post
title: "Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: Shenhe.png
---

# Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language
# ZSL

## 摘要
学习从训练数据中未包含的类别中对视频数据进行分类，即基于视频的零样本学习。

* 本论文认为视频数据中音频和视觉模态之间的自然对齐为学习有区别的多模态表示提供了丰富的训练信号。
* 专注于视听零镜头学习的相对未充分探索的任务，本论文建议使用跨模态注意力从视听数据中学习多模态表示，并利用文本标签嵌入将知识从可见类转移到不可见类类。
* 更进一步，在本论文的广义视听零样本学习设置中，他们在测试时的搜索空间加入训练的类，这些训练类充当干扰物并增加难度。
* 由于该领域缺乏统一的基准，我们在三个不同大小和难度的视听数据集 VGGSound、UCF 和 ActivityNet 上引入（广义）零样本学习基准，确保看不见的测试类不会出现在用于骨干深度模型的监督训练的数据集中。

