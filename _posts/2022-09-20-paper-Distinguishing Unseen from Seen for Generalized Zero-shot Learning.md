---
layout: post
title: "Distinguishing Unseen from Seen for Generalized Zero-shot Learning"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: Cha1.png

---

# Distinguishing Unseen from Seen for Generalized Zero-shot Learning

# GZSL

## 摘要

### 广义零样本学习（GZSL）旨在识别其类别可能在训练中没有出现过的样本。

### 老方法：会将未见类识别为已见类，反之亦然。

### 老方法问题：通常会导致 GZSL 的性能不佳。

### <u>ZSL本质：从可见类中学习得到的视觉表示和语义描述，映射到一个共享空间，我们在该空间中来做未见类的分类。</u>

### <u>GZSL与ZSL区别：GZSL的测试集包括训练集与测试集</u>

### 解决方案：

+ 在本文中，我们提出了一种利用视觉和语义模式来区分可见和不可见类别的新方法。
+ 具体来说，我们的方法部署了两个**变分自动编码器**，以在**共享的潜在空间中生成视觉和语义模态的潜在表示**，其中我们**通过 Wasserstein 距离对齐两种模态的潜在表示**，**并用彼此的表示重建两种模态**。
+ **为了在可见和不可见类别之间学习更清晰的边界**，我们提出了一种两阶段训练策略，**该策略利用可见和不可见的语义描述并搜索阈值以分离可见和不可见的视觉样本。**
+ 最后，使用见过的专家和未见过的专家进行最终分类。

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209201634986.png)



### 怎么做：

+ As illustrated in Figure 2, our method learns two variational auto-encoders, one for each modality for zero-shot learning. The training process can be split into two stages. 
  + At the first stage, we map visual features and semantic descriptions into a shared latent space, in which we align the latent representations of the two modalities and learn a classifier for seen classes. After this stage, the representations encoded by our model is expected to be modality-invariant for these seen classes. 
  + At the second stage, we explicitly leverage the semantic descriptions of unseen classes to synthesize artificial samples that form the *fictitious class* in the visual space, where we exploit the modality-invariant features for unseen classes as in the first stage. We deploy another classifier to separate fictitious classes and seen classes in the latent space. After the two-stage training, latent representations of both modalities are aligned class-wisely and a boundary of each class can be easily found. By analyzing these boundaries of seen categories, we compute a threshold that can separate seen and unseen samples. Once seen and unseen classes are separated, arbitrary seen or unseen experts can be adopted to carry out visual classification.

+  Specifically,  we  deploy  hypersphericalVAE [13] models for both visual and semantic modalitiesand  align  the  latent  representations  of  the  two  modalitiesat the category-level.  To leverage fictitious class, we propose a two-stage training scheme.   
  + Specifically,  we firstly train both visual and semantic VAE models with seen sam-ples and corresponding semantic attributes. 
  + Then we generate fictitious classes and train semantic VAE with fictitioussamples and unseen semantic attributes. 
  +  We measure similarity between the latent representations of two modalitiesand search a threshold to distinguish seen and unseen do-mains.  
  + By this, seen and unseen samples can be success-fully distinguished.  Further, we propose an unseen expertwhich is regularized by attention mechanism to classify un-seen visual samples.


### 贡献：

To summarize, the main contributions of this paper arethreefold: 

+  (1)  We  propose  a  novel  method  to  distinguishseen and unseen domains for GZSL. We design a two-stagetraining  scheme  which  significantly  improves  the  modelperformance  by  leveraging  both  seen  and  unseen  seman-tic attributes. 
+  (2) We propose to leverage a novel fictitiousclass to separate similar visual representations.  With ficti-tious class, we can successfully separate indistinguishableseen and unseen samples.  In addition, we propose an un-seen expert with attention mechanism to recognize unseensamples.  It is worth noting that the unseen expert can betrained less than a minute in all tested datasets. 
+ (3) We con-duct extensive experiments on five open benchmarks.  Theresults verify that the proposed method can significantly im-prove the result of previous state-of-the-art approaches



