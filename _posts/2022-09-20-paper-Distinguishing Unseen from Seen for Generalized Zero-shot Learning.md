---
layout: post
title: "Distinguishing Unseen from Seen for Generalized Zero-shot Learning"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: scene3.jpeg

---

# Distinguishing Unseen from Seen for Generalized Zero-shot Learning

# ZSL

## 摘要

### 多标签零样本学习（ZSL）是标准单标签 ZSL 的更现实的部分，因为多个对象可以在自然图像中共存。然而，多个对象的出现使推理复杂化，并且需要对视觉特征进行特定区域的处理以保留其上下文线索。

### 老方法：我们注意到，现有的最佳多类别 ZSL 方法采用共享方法来关注区域特征，并为所有类提供一组共同的注意力图。

### 老方法问题：这种共享地图会导致注意力分散，当类的数量很大时，它不会有区别地关注相关位置。此外，将空间池化的视觉特征映射到类语义会导致类间特征纠缠，从而阻碍分类。

### 解决方案：

+ 在这里，我们提出了一种基于区域的可辨别性保留多标签零样本分类的替代方法。
+ 我们的方法保持空间分辨率以保留区域级特征，并利用双层注意模块（BiAM）通过结合区域和场景上下文信息来丰富特征。
+ 然后将丰富的区域级特征映射到类语义，并且只有它们的类预测被空间池化以获得图像级预测，从而保持多类特征解开。



![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201543572.png)

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201544395.png)

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201544012.png)



### 贡献：

+ 注意力机制！！！这篇文章的注意力机制很好，很好的提取了特征

### 怎么做：

+ 未完待续

