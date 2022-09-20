---
layout: post
title: "Counterfactual Zero-Shot and Open-Set Visual Recognition"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: scene1.jpeg

---

# Counterfactual Zero-Shot and Open-Set Visual Recognition

# ZSL

## 摘要

### 我们为零样本学习 (ZSL) 和开放集识别 (OSR) 提出了一个新颖的反事实框架，其共同挑战是仅通过对可见类进行训练来推广到未见类。

### 问题：我们的想法源于观察到为未见类生成的样本通常超出真实分布，这导致已见类（高）和未见类（低）之间的识别率严重失衡。

### 问题的原因：我们表明，关键原因是生成模型不是（反事实忠实）的，因此我们提出了一个忠实的生成模型。

### 解决方案：

+ 设置一个样本为某个类，同时保持其样本属性不变。
+ 由于忠实性，我们可以应用一致性规则来执行看不见/看不见的二元分类。
+ 通过询问：它的反事实是否仍然看起来像它自己？如果“是”，则样本来自某个类别，否则为“否”。



![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201452571.png)



### 怎么做：

+ 这篇到底在干嘛我属实是没看懂，后面再说吧^^_



