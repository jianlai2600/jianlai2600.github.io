---
layout: post
title: "Contrastive Embedding for Generalized Zero-Shot Learning"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: Ganyu2.jpeg

---

# Contrastive Embedding for Generalized Zero-Shot Learning

# ZSL

## 摘要

### **广义零样本学习（GZSL）旨在当仅提供来自可见类的标签示例时，模型可以识别来自可见和不可见类的对象。**

### 老方法：最近的特征生成方法学习了一种生成模型，该模型可以合成看不见的类别的缺失视觉特征，以缓解 GZSL 中的数据不平衡问题。

### 老方法的问题： 然而，原始视觉特征空间对于 GZSL 分类来说是次优的，因为它缺乏判别信息。

### **解决方案：**

* 将生成模型与嵌入模型相结合，产生一个混合 GZSL 框架。混合 GZSL 方法绘制了生成的真实样本和合成样本。将模型放入一个嵌入空间，在这里我们执行最终的 GZSL 分类。
* 具体来说，我们为我们的混合 GZSL 框架提出了一种对比嵌入 (CE)。所提出的对比嵌入不仅可以利用类监督，还可以利用**实例监督**，后者通常被现有的 GZSL 研究忽略。

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201407572.png)



### 怎么做：

+ We learn an embedding function E that **maps the visual samples xi into the embedding space as hi = E(xi).** 
+ We further learn a **non-linear projection H to better constrain the embedding space: zi = H(hi).** 
+ We introduce a **comparator network F that measures the relevance score between hi and the semantic descriptors.** 
+ We **learn the embedding function with both the instance-level and the class-level supervisions.** 
+ We **integrate the contrastive embedding model with the feature generation model.** 
+ In the feature generation model, **the feature generator G learns to produce visual features based on a semantic descriptor a and a Gaussian noise ε; and the discriminator D aims to distinguish the fake visual features from real ones**.



