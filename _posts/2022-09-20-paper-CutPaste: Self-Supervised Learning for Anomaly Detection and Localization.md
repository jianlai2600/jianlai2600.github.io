---
layout: post
title: "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: scene1.jpeg

---

# CutPaste: Self-Supervised Learning for Anomaly Detection and Localization

# ZSL

## 摘要

### 我们的目标是构建一个高性能的缺陷检测模型，在没有异常数据的情况下检测图像的未知异常模式。为此，我们提出了一个仅使用正常训练数据构建异常检测器的两阶段框架。



### 解决方案：

+ 我们首先学习自我监督的深度表示，然后在学习的表示上构建一个生成的一类分类器。
+ 我们通过对来自 CutPaste 的正常数据进行分类来学习表示，这是一种简单的数据增强策略，可以剪切图像补丁并粘贴到大图像的随机位置。
+ 我们对 MVTec 异常检测数据集的实证研究表明，所提出的算法通常能够检测各种类型的现实世界缺陷。
+ 在从头开始学习表示时，我们通过 3.1 AUC 对以前的艺术进行了改进。
+ 通过对 ImageNet 上预训练表示的迁移学习，我们实现了新的最先进的 96.6 AUC。
+ 最后，我们扩展了框架以从补丁中学习和提取表示，以允许在训练期间定位缺陷区域而无需注释。



<img src="https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201515103.png" style="zoom:80%;" />



![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201516140.png)



### 怎么做：

+ 未完待续

