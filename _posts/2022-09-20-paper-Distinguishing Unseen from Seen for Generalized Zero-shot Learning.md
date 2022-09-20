---
layout: post
title: "Distinguishing Unseen from Seen for Generalized Zero-shot Learning"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: Cha1.png

---

# Distinguishing Unseen from Seen for Generalized Zero-shot Learning

# ZSL

## 摘要

### 广义零样本学习（GZSL）旨在识别其类别可能在训练中没有出现过的样本。

### 老方法：将未见类识别为已见类，反之亦然。

### 老方法问题：通常会导致 GZSL 的性能不佳。

### 解决方案：

+ 在本文中，我们提出了一种利用视觉和语义模式来区分可见和不可见类别的新方法。
+ 具体来说，我们的方法部署了两个**变分自动编码器**，以在**共享的潜在空间中生成视觉和语义模态的潜在表示**，其中我们**通过 Wasserstein 距离对齐两种模态的潜在表示**，**并用彼此的表示重建两种模态**。
+ **为了在可见和不可见类别之间学习更清晰的边界**，我们提出了一种两阶段训练策略，**该策略利用可见和不可见的语义描述并搜索阈值以分离可见和不可见的视觉样本。**
+ 最后，使用见过的专家和未见过的专家进行最终分类。

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201634986.png)

### 怎么做：

+ 未完待续

### 贡献：

+ 



