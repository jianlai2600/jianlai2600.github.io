---
layout: post
title: "2020-2022 CVPR OCR With Code"
author: "Yafei Li"
categories: paper
tags: [papers, ZSL]
image: Cha1.png

---

# 2020-2022 CVPR OCR With Code

# OCR

### Template👇

### <u>Scene Text Detection</u>

#### Abstract



#### Idea

Old method:	

Old method flaws:	

**<u>*New method:*</u>**	

#### Figure



#### Process



# *Text Detection*

### <u>ContourNet: Taking a Further Step toward Accurate Arbitrary-shaped Scene Text Detection</u>

#### Abstract

*Scene text detection has witnessed rapid development in recent years. However, there still exists two main chal- lenges: 1) many methods suffer from false positives in their text representations; 2) the large scale variance of scene texts makes it hard for network to learn samples. In this paper, we propose the ContourNet, which effectively han- dles these two problems taking a further step toward ac- curate arbitrary-shaped text detection. At first, a scale- insensitive Adaptive Region Proposal Network (Adaptive- RPN) is proposed to generate text proposals by only focus- ing on the Intersection over Union (IoU) values between predicted and ground-truth bounding boxes. Then a nov- el Local Orthogonal Texture-aware Module (LOTM) mod- els the local texture information of proposal features in t- wo orthogonal directions and represents text region with a set of contour points. Considering that the strong unidi- rectional or weakly orthogonal activation is usually caused by the monotonous texture characteristic of false-positive patterns (e.g*. *streaks.), our method effectively suppresses these false positives by only outputting predictions with high response value in both orthogonal directions. This gives more accurate description of text regions. Extensive exper- iments on three challenging datasets (Total-Text, CTW1500 and ICDAR2015) verify that our method achieves the state- of-the-art performance.*

#### Idea

Old method flaws:	

+ false positive
+ large scale variance of scene texts makes it hard for network to learn samples

**<u>*New method:	Propose a ContourNet to solve above problems*</u>**

+ At first, a scale- insensitive Adaptive Region Proposal Network (Adaptive- RPN) is proposed to generate text proposals by only focusing on the Intersection over Union (IoU) values between predicted and ground-truth bounding boxes. 

**<u>*Effectively propose text proposal region*</u>**

+ Then a novel Local Orthogonal Texture-aware Module (LOTM) models the local texture information of proposal features in two orthogonal directions and represents text region with a set of contour points. 

**<u>*Model the feature in text proposal region，get feature in a better way，represents text region with a set of contour points*</u>**

+ Considering that the strong unidi- rectional or weakly orthogonal activation is usually caused by the monotonous texture characteristic of false-positive patterns (e.g*. *streaks.), our method effectively suppresses these false positives by only outputting predictions with high response value in both orthogonal directions. This gives more accurate description of text regions.

**<u>*Better prediction，decrease false positive*</u>**

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231735417.png)

#### Process



### <u>Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection</u>

#### Abstract

*Arbitrary shape text detection is a challenging task due to the high variety and complexity of scenes texts. In this pa- per, we propose a novel unified relational reasoning graph network for arbitrary shape text detection. In our method, an innovative local graph bridges a text proposal model via Convolutional Neural Network (CNN) and a deep re- lational reasoning network via Graph Convolutional Net- work (GCN), making our network end-to-end trainable. To be concrete, every text instance will be divided into a se- ries of small rectangular components, and the geometry at- tributes (*e.g*., height, width, and orientation) of the small components will be estimated by our text proposal model. Given the geometry attributes, the local graph construc- tion model can roughly establish linkages between differ- ent text components. For further reasoning and deduc- ing the likelihood of linkages between the component and its neighbors, we adopt a graph-based network to per- form deep relational reasoning on local graphs. Experi- ments on public available datasets demonstrate the state- of-the-art performance of our method.*

#### Idea

**<u>*New method:*</u>**	In our method, an innovative local graph bridges **<u>*a text proposal model via Convolutional Neural Network (CNN)*</u>** and **<u>*a deep relational reasoning network via Graph Convolutional Network (GCN)*</u>**, making our network end-to-end trainable. 

**<u>*We propose a novel unified relational reasoning graph network for arbitrary shape text detection,and we can train in a e2e way.*</u>**

+ Every text instance will be divided into a series of small rectangular components, and the geometry attributes (e.g., height, width, and orientation) of the small components will be estimated by our text proposal model. 
+ Given the geometry attributes, the local graph construction model can roughly establish linkages between differ- ent text components. 
+ For further reasoning and deducing the likelihood of linkages between the component and its neighbors, we adopt a graph-based network to perform deep relational reasoning on local graphs.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209242132759.png)

#### Process



# *Text Recognition*

### <u>A Multiplexed Network for End-to-End, Multilingual OCR</u>

#### Abstract

*Recent advances in OCR have shown that an end-to- end (E2E) training pipeline that includes both detection and recognition leads to the best results. However, many exist- ing methods focus primarily on Latin-alphabet languages, often even only case-insensitive English characters. In this paper, we propose an E2E approach, Multiplexed Multilin- gual Mask TextSpotter, that performs script identification at the word level and handles different scripts with differ- ent recognition heads, all while maintaining a unified loss that simultaneously optimizes script identification and mul- tiple recognition heads. Experiments show that our method outperforms the single-head model with similar number of parameters in end-to-end recognition tasks, and achieves state-of-the-art results on MLT17 and MLT19 joint text de- tection and script identification benchmarks. We believe that our work is a step towards the end-to-end trainable and scalable multilingual multi-purpose OCR system. Our code and model will be released.*

#### Idea

Old method:	Focus primarily on Latin-alphabet languages, often even only case-insensitive English characters

<u>**New method：E2E detection & recognition framework.The proposed M3 TextSpotter shares the same detection and segmentation trunk with Mask TextSpotter v3 , but incorporates a novel Language Prediction Network (LPN). The output of the LPN then determines which script’s recognition head the multiplexer selects.**</u>



+ performs script identification at the word level and handles different scripts with different recognition heads

+ all while maintaining a unified loss that simultaneously optimizes script identification and multiple recognition heads. 

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231644340.png)

#### Process



### <u>Dictionary-guided Scene Text Recognition</u>

#### Abstract

**<u>Language prior plays an important role in the way humans detect and recognize text in the wild</u>**. Current scene text recognition methods do use lexicons to improve recog- nition performance, but their naive approach of casting the output into a dictionary word based purely on the edit dis- tance has many limitations. In this paper, we present a novel approach to incorporate a dictionary in both the training and inference stage of a scene text recognition system. We use the dictionary to generate a list of possible outcomes and find the one that is most compatible with the visual ap- pearance of the text. The proposed method leads to a ro- bust scene text recognition model, which is better at han- dling ambiguous cases encountered in the wild, and im- proves the overall performance of state-of-the-art scene text spotting frameworks. Our work suggests that incorporating language prior is a potential approach to advance scene text detection and recognition methods. Besides, we con- tribute VinText, a challenging scene text dataset for Viet- namese, where some characters are equivocal in the vi- sual form due to accent symbols. This dataset will serve as a challenging benchmark for measuring the applicabil- ity and robustness of scene text detection and recognition algorithms.

#### Idea

Old method:	Current scene text recognition methods do use lexicons to improve recognition performance,

Old method flaws:	But their naive approach of casting the output into a dictionary word based purely on the edit distance has many limitations. 

**<u>*New method:	In this paper, we present a novel approach to incorporate a dictionary in both the training and inference stage of a scene text recognition system.*</u>** 

+ We use the dictionary to **<u>*generate a list of possible outcomes and find the one that is most compatible*</u>** with the visual appearance of the text. 
+ The proposed method leads to a robust scene text recognition model, which is better at handling ambiguous cases encountered in the wild, and improves the overall performance of state-of-the-art scene text spotting frameworks. 
+ Our work suggests that **<u>*incorporating language prior*</u>** is a potential approach to advance scene text detection and recognition methods. 

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209242151312.png)

#### Process



### <u>Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition</u>

#### Abstract

*Handwritten text and scene text suffer from various shapes and distorted patterns. Thus training a robust recognition model requires a large amount of data to cover diversity as much as possible. In contrast to data collection and annotation, data augmentation is a low cost way. In this paper, we propose a new method for text image augmentation. Different from traditional augmentation methods such as rotation, scaling and perspective transformation, our pro- posed augmentation method is designed to learn proper and efficient data augmentation which is more effective and spe- cific for training a robust recognizer. By using a set of cus- tom fiducial points, the proposed augmentation method is flexible and controllable. Furthermore, we bridge the gap between the isolated processes of data augmentation and network optimization by joint learning. An agent network learns from the output of the recognition network and con- trols the fiducial points to generate more proper training samples for the recognition network. Extensive experiments on various benchmarks, including regular scene text, irreg- ular scene text and handwritten text, show that the proposed augmentation and the joint learning methods significantly boost the performance of the recognition networks. A gen- eral toolkit for geometric augmentation is available*1*.*

#### Idea

Old method:	Handwritten text and scene text suffer from various shapes and distorted patterns.

Old method flaws:	Thus training a robust recognition model requires a large amount of data to cover diversity as much as possible.

**<u>*New method:	Data augmentation*</u>** is a low cost way.

+ We propose **<u>*a new method for text image augmentation*</u>**, our proposed augmentation method is designed to learn proper and efficient data augmentation which is more effective and specific for training a robust recognizer. 
+ By using a set of **<u>*custom fiducial points*</u>**, the proposed augmentation method is flexible and controllable. 
+ Furthermore, we **<u>*bridge the gap between the isolated processes of data augmentation and network optimization by joint learning.*</u>** 
+ An **<u>*agent network*</u>** learns from the output of the recognition network and **<u>*controls the fiducial points to generate more proper training samples*</u>** for the recognition network. 
+ Extensive experiments on various benchmarks, including regular scene text, irreg- ular scene text and handwritten text, show that the proposed augmentation and the joint learning methods significantly boost the performance of the recognition networks. 
+ A general toolkit for geometric augmentation is available*1*.

#### Figure



#### Process



# *Text Spotting*

### <u>ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network</u>

#### Abstract

*Scene text detection and recognition has received increasing research attention. Existing methods can be roughly categorized into two groups: character-based and segmentation-based. These methods either are cost- ly for character annotation or need to maintain a complex pipeline, which is often not suitable for real-time applica- tions. Here we address the problem by proposing the Adap- tive Bezier-Curve Network (ABCNet). Our contributions are three-fold: 1) For the first time, we adaptively fit ori- ented or curved text by a parameterized Bezier curve. 2) We design a novel BezierAlign layer for extracting accu- rate convolution features of a text instance with arbitrary shapes, significantly improving the precision compared with previous methods. 3) Compared with standard bounding box detection, our Bezier curve detection introduces negli- gible computation overhead, resulting in superiority of our method in both efficiency and accuracy.*

*Experiments on oriented or curved benchmark datasets, namely Total-Text and CTW1500, demonstrate that ABCNet achieves state-of-the-art accuracy, meanwhile significantly improving the speed. In particular, on Total-Text, our real- time version is over 10 times faster than recent state-of-the- art methods with a competitive recognition accuracy.*

#### Idea

Old method:	Existing methods can be roughly categorized into two groups: character-based and segmentation-based.

Old method flaws:	These methods either are costly for character annotation or need to maintain a complex pipeline, which is often not suitable for real-time applications.

**<u>*New method:	Propose ABCNet*</u>**

+ For the first time, we adaptively fit oriented or curved text by a parameterized Bezier curve. 
+ We design a novel BezierAlign layer for extracting accurate convolution features of a text instance with arbitrary shapes, significantly improving the precision compared with previous methods. 
+ Compared with standard bounding box detection, our Bezier curve detection introduces negligible computation overhead, resulting in superiority of our method in both efficiency and accuracy.

***<u>Fits the text boxes well，gets feature in a better way</u>***

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231708624.png)

#### Process



### <u>Implicit Feature Alignment: Learn to Convert Text Recognizer to Text Spotter</u>

#### Abstract

*Text recognition is a popular research subject with many associated challenges. Despite the considerable progress made in recent years, the text recognition task itself is still constrained to solve the problem of reading cropped line text images and serves as a subtask of optical character recognition (OCR) systems. As a result, the final text recog- nition result is limited by the performance of the text de- tector. In this paper, we propose a simple, elegant and ef- fective paradigm called Implicit Feature Alignment (IFA), which can be easily integrated into current text recogniz- ers, resulting in a novel inference mechanism called IFA-inference. This enables an ordinary text recognizer to pro- cess multi-line text such that text detection can be com- pletely freed. Specifically, we integrate IFA into the two most prevailing text recognition streams (attention-based and CTC-based) and propose attention-guided dense pre- diction (ADP) and Extended CTC (ExCTC). Furthermore, the Wasserstein-based Hollow Aggregation Cross-Entropy (WH-ACE) is proposed to suppress negative predictions to assist in training ADP and ExCTC. We experimentally demonstrate that IFA achieves state-of-the-art performance on end-to-end document recognition tasks while maintaining the fastest speed, and ADP and ExCTC complement each other on the perspective of different application scenarios.*

#### Idea

Old method:Despite the considerable progress made in recent years, the text recognition task itself is still constrained to solve the problem of reading cropped line text images and serves as a subtask of optical character recognition (OCR) systems.

Old method flaws:As a result, the final text recognition result is limited by the performance of the text detector.

**<u>*New method:*</u>**

+ We propose a simple, elegant and effective paradigm called Implicit Feature Alignment (IFA), which can be easily integrated into current text recognizers, resulting in a novel inference mechanism called IFA-inference. 
+ This enables an ordinary text recognizer to process multi-line text such that text detection can be completely freed. 
+ Specifically, we integrate IFA into the two most prevailing text recognition streams (attention-based and CTC-based) and propose attention-guided dense prediction (ADP) and Extended CTC (ExCTC). 
+ Furthermore, the Wasserstein-based Hollow Aggregation Cross-Entropy (WH-ACE) is proposed to suppress negative predictions to assist in training ADP and ExCTC. 
+ We experimentally demonstrate that IFA achieves state-of-the-art performance on end-to-end document recognition tasks while maintaining the fastest speed, and ADP and ExCTC complement each other on the perspective of different application scenarios.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209242228614.png)

#### Process































