---
layout: post
title: "2021-2022 CVPR OCR"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: Cha1.png

---

# 2021-2022 CVPR OCR

# OCR

### Template👇

### <u>Scene Text Detection</u>

#### Abstract

#### Idea

#### Figure

#### Architecture

# *Text Detection*

### <u>Few Could Be Better Than All:Feature Sampling and Grouping for Scene Text Detection</u>

#### Abstract

*Recently, transformer-based methods have achieved promising progresses in object detection, as they can elim- inate the post-processes like NMS and enrich the deep rep- resentations. However, these methods cannot well cope with scene text due to its extreme variance of scales and aspect ratios. In this paper, we present a simple yet ef- fective transformer-based architecture for scene text detec- tion. Different from previous approaches that learn robust deep representations of scene text in a holistic manner, our method performs scene text detection based on a few rep- resentative features, which avoids the disturbance by back- ground and reduces the computational cost. Specifically, we first select a few representative features at all scales that are highly relevant to foreground text. Then, we adopt a transformer for modeling the relationship of the sam- pled features, which effectively divides them into reason- able groups. As each feature group corresponds to a text in- stance, its bounding box can be easily obtained without any post-processing operation. Using the basic feature pyra- mid network for feature extraction, our method consistently achieves state-of-the-art results on several popular datasets for scene text detection.*

#### Idea

简单有效的基于Transformer的场景文本检测框架

~~以往整体的学习场景文本的鲁棒深度表示~~  -> 新方法基于一些代表性特征执行场景文本检测，避免了背景的干扰并降低了计算成本

做法：

+ 具体来说，我们首先在所有尺度上选择一些与前景文本高度相关的代表性特征。
+ 然后，我们采用变换器对采样特征的关系进行建模，有效地将它们划分为合理的组。
+ 由于每个特征组对应一个文本实例，因此无需任何后处理操作即可轻松获得其边界框。
+ 使用基本特征金字塔网络进行特征提取，我们的方法在用于场景文本检测的几个流行数据集上始终如一地实现了最先进的结果。

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231120644.png)

#### Architecture

+ It consists of a backbone network, a multi-scale feature sampling network, and a feature grouping network. 
+ Specifically, multi-scale feature maps are first produced from the backbone network. 
+ Next, a multi-scale text extractor is used to predict the confidence scores of the representative text regions at the pixel level. 
+ Then, we select text point features with top-N scores and concatenate them with position embeddings. 
+ After that, we adopt a transformer to model the relationship between the sampled features and implicitly group them into fine representations by the attention mechanism. 
+ Finally, the detection results are obtained from the prediction heads.



# *Text Recognition*

### <u>Open-Set Text Recognition via Character-Context Decoupling</u>

#### Abstract

*The open-set text recognition task is an emerging challenge that requires an extra capability to cognize novel characters during evaluation. We argue that a major cause of the limited performance for current methods is the confounding effect of contextual information over the visual information of individual characters. Under open-set scenarios, the intractable bias in contextual information can be passed down to visual information, consequently impairing the classification performance. In this paper, a Character-Context Decoupling framework is proposed to alleviate this problem by separating contextual informa- tion and character-visual information. Contextual informa- tion can be decomposed into temporal information and linguistic information. Here, temporal information that mod- els character order and word length is isolated with a detached temporal attention module. Linguistic information that models n-gram and other linguistic statistics is separated with a decoupled context anchor mechanism. A va- riety of quantitative and qualitative experiments show that our method achieves promising performance on open-set, zero-shot, and close-set text recognition datasets.*

#### Idea

open-set->识别新的字

老方法问题：上下文信息（中的bias）对单个字符信息的混淆作用，影响单个字符的分类

**上下文信息可以被分解为时序信息和语言信息**

***Temporal information*** -> *models character order and word length* ->*isolated with a **detached temporal attention module***

***Linguistic information*** ->*models **n-gram** and other linguistic statistics* -> *separated with a **decoupled context anchor mechanism***

**[术语]n-gram -> 预计或者评估一个句子是否合理**



**<u>*该方法的妙就是将context与字符解藕，专注于单个字的salience region，而老方法会去看context，而context会有坏影响*</u>**

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231427956.png)

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231431394.png)

#### Architecture

+ In the framework, visual representation of the sample and character templates are first extracted with the DSBN-Res45 Network
+ Then the Detached Temporal Attention module predicts the word length and samples visual features x[t] for each timestamp. 
+ The visual prediction is achieved by matching prototypes (attention-reduced template features) with the Open-set classifier. 
+ Finally, the visual prediction is adjusted with the Decoupled Context Anchor module, and no adjustment is conducted when there is intractable linguistic information under open-set scenarios.



### <u>Pushing the Performance Limit of Scene Text Recognizer without Human Annotation</u>

#### Abstract

*Scene text recognition (STR) attracts much attention over the years because of its wide application. Most methods train STR model in a fully supervised manner which re- quires large amounts of labeled data. Although synthetic data contributes a lot to STR, it suffers from the real-to- synthetic domain gap that restricts model performance. In this work, we aim to boost STR models by leveraging both synthetic data and the numerous real unlabeled images, ex- empting human annotation cost thoroughly. A robust con- sistency regularization based semi-supervised framework is proposed for STR, which can effectively solve the instabil- ity issue due to domain inconsistency between synthetic and real images. A character-level consistency regularization is designed to mitigate the misalignment between characters in sequence recognition. Extensive experiments on stan- dard text recognition benchmarks demonstrate the effective- ness of the proposed method. It can steadily improve exist- ing STR models, and boost an STR model to achieve new state-of-the-art results. To our best knowledge, this is the first consistency regularization based framework that ap- plies successfully to STR.*

#### Idea

STR依赖于标注数据然后监督学习，很繁琐

老方法问题：合成数据集尽管很好，但是受限于合成-真实之间的差距，限制了模型的性能

新方法：使用合成数据和未标注真实数据

<u>**A robust consistency regularization based semi-supervised framework** is proposed for STR, which can effectively solve the instability issue due to domain inconsistency between synthetic and real images.</u>

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231510041.png)

#### Architecture

+ <u>take advantage of labeled synthetic data and unlabeled real data</u>
+ An asymmetric structure is designed with EMA and domain adaption to encourage a stable model training



### <u>SimAN: Exploring Self-Supervised Representation Learning of Scene Text via Similarity-Aware Normalization</u>

#### Abstract

*Recently self-supervised representation learning has drawn considerable attention from the scene text recogni- tion community. Different from previous studies using con- trastive learning, we tackle the issue from an alternative perspective, i.e., by formulating the representation learning scheme in a generative manner. Typically, the neighbor- ing image patches among one text line tend to have simi- lar styles, including the strokes, textures, colors, etc. Moti- vated by this common sense, we augment one image patch and use its neighboring patch as guidance to recover itself. Specifically, we propose a Similarity-Aware Normalization (SimAN) module to identify the different patterns and align the corresponding styles from the guiding patch. In this way, the network gains representation capability for distin- guishing complex patterns such as messy strokes and clut- tered backgrounds. Experiments show that the proposed SimAN significantly improves the representation quality and achieves promising performance. Moreover, we surpris- ingly find that our self-supervised generative network has impressive potential for data synthesis, text image editing, and font interpolation, which suggests that the proposed SimAN has a wide range of practical applications.*

#### Idea

#### Figure

#### Architecture



