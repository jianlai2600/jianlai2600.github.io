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

**'''**

### <u>Scene Text Detection</u>

#### Abstract



#### Idea

Old method:	

Old method flaws:	

**<u>*New method:*</u>**	

#### Figure



#### Process

**'''**

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



### <u>Progressive Contour Regression for Arbitrary-Shape Scene Text Detection</u>

#### Abstract

*State-of-the-art scene text detection methods usually model the text instance with local pixels or components from the bottom-up perspective and, therefore, are sensitive to noises and dependent on the complicated heuristic post-processing especially for arbitrary-shape texts. To relieve these two issues, instead, we propose to progressive- ly evolve the initial text proposal to arbitrarily shaped text contours in a top-down manner. The initial horizontal text proposals are generated by estimating the center and size of texts. To reduce the range of regression, the first stage of the evolution predicts the corner points of oriented tex- t proposals from the initial horizontal ones. In the second stage, the contours of the oriented text proposals are itera- tively regressed to arbitrarily shaped ones. In the last iter- ation of this stage, we rescore the confidence of the final localized text by utilizing the cues from multiple contour points, rather than the single cue from the initial horizon- tal proposal center that may be out of arbitrary-shape text regions. Moreover, to facilitate the progressive contour evo- lution, we design a contour information aggregation mech- anism to enrich the feature representation on text contours by considering both the circular topology and semantic con- text. Experiments conducted on CTW1500, Total-Text, ArT, and TD500 have demonstrated that the proposed method e- specially excels in line-level arbitrary-shape texts.*

#### Idea

Old method:	Model the text instance with local pixels or components from the bottom-up perspective

Old method flaws:	Sensitive to noises and dependent on the complicated heuristic post-processing especially for arbitrary-shape texts

**<u>*New method:	We propose to progressively evolve the initial text proposal to arbitrarily shaped text contours in a top-down manner.*</u>** 

+ The initial horizontal text proposals are generated by estimating the center and size of texts. 
+ To reduce the range of regression, the first stage of the evolution predicts the corner points of oriented text proposals from the initial horizontal ones. 
+ In the second stage, the contours of the oriented text proposals are iteratively regressed to arbitrarily shaped ones. 
+ In the last iteration of this stage, we rescore the confidence of the final localized text by utilizing the cues from multiple contour points, rather than the single cue from the initial horizontal proposal center that may be out of arbitrary-shape text regions. 
+ Moreover, to facilitate the progressive contour evolution, we design a contour information aggregation mechanism to enrich the feature representation on text contours by considering both the circular topology and semantic context.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209251710966.png)

#### Process



### <u>Towards End-to-End Unified Scene Text Detection and Layout Analysis</u>

#### Abstract

*Scene text detection and document layout analysis have long been treated as two separate tasks in different image domains. In this paper, we bring them together and introduce the task of unified scene text detection and layout analysis. The first hierarchical scene text dataset is introduced to enable this novel research task. We also propose a novel method that is able to simultaneously de- tect scene text and form text clusters in a unified way. Comprehensive experiments show that our unified model achieves better performance than multiple well-designed baseline methods. Additionally, this model achieves state- of-the-art results on multiple scene text detection datasets without the need of complex post-processing.*

#### Idea

Old method:	Scene text detection and document layout analysis 

Old method flaws:	Scene text detection and document layout analysis have long been treated as two **<u>*separate tasks*</u>** in different image domains.

**<u>*New method:	In this paper, we bring them together and introduce the task of unified scene text detection and layout analysis.*</u>**

+ The first hierarchical scene text dataset is introduced to enable this novel research task. 
+ <u>***We also propose a novel method that is able to simultaneously detect scene text and form text clusters in a unified way.***</u> 

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209261613586.png)

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
+ A general **<u>*toolkit for geometric augmentation*</u>** is available*1*.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209242257241.png)

#### Process



### <u>Open-Set Text Recognition via Character-Context Decoupling</u>

#### Abstract

*The open-set text recognition task is an emerging challenge that requires an extra capability to cognize novel characters during evaluation. We argue that a major cause of the limited performance for current methods is the confounding effect of contextual information over the visual information of individual characters. Under open-set scenarios, the intractable bias in contextual information can be passed down to visual information, consequently impairing the classification performance. In this paper, a Character-Context Decoupling framework is proposed to alleviate this problem by separating contextual informa- tion and character-visual information. Contextual informa- tion can be decomposed into temporal information and linguistic information. Here, temporal information that mod- els character order and word length is isolated with a de- tached temporal attention module. Linguistic information that models n-gram and other linguistic statistics is sepa- rated with a decoupled context anchor mechanism. A va- riety of quantitative and qualitative experiments show that our method achieves promising performance on open-set, zero-shot, and close-set text recognition datasets.*

#### Idea

Old method:	The open-set text recognition task is an emerging challenge that requires an extra capability to cognize novel characters during evaluation. 

Old method flaws:	A major cause of the limited performance for current methods is the confounding effect of contextual information over the visual information of individual characters. Under open-set scenarios, the intractable bias in contextual information can be passed down to visual information, consequently impairing the classification performance.

**<u>*New method:*</u>**	

+ A Character-Context Decoupling framework is proposed to alleviate this problem by separating contextual information and character-visual information. 
+ Contextual information can be decomposed into **<u>*temporal information and linguistic information*</u>**. 
+ Here, **<u>*temporal information*</u>** that models character order and word length is isolated with a detached temporal attention module. 
+ **<u>*Linguistic information*</u>** that models n-gram and other linguistic statistics is separated with a decoupled context anchor mechanism. 

**<u>*Focus on decoupling the context and the character,the context have bad influence*</u>**

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209231431394.png)

#### Process



### <u>OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page Text Recognition by learning to unfold</u>

#### Abstract

*Text recognition is a major computer vision task with a big set of associated challenges. One of those traditional challenges is the coupled nature of text recognition and segmentation. This problem has been progressively solved over the past decades, going from segmentation based recognition to segmentation free approaches, which proved more accurate and much cheaper to annotate data for. We take a step from segmentation-free single line recognition towards segmentation-free multi-line / full page recognition. We propose a novel and simple neural network module, termed* *OrigamiNet**, that can augment any CTC-trained, fully con- volutional single line text recognizer, to convert it into a multi-line version by providing the model with enough spa- tial capacity to be able to properly collapse a 2D input sig- nal into 1D without losing information. Such modified net- works can be trained using exactly their same simple origi- nal procedure, and using only* *unsegmented* *image and text pairs. We carry out a set of interpretability experiments that show that our trained models learn an accurate im- plicit line segmentation. We achieve state-of-the-art char- acter error rate on both IAM & ICDAR 2017 HTR bench- marks for handwriting recognition, surpassing all other methods in the literature. On IAM we even surpass sin- gle line methods that use accurate localization information during training.*

#### Idea

Old method:	Segmentation based recognition

Old method flaws:	High costs for annotation

**<u>*New method:	Segmentation free approaches,multi-line recognition*</u>**

+ We propose a novel and simple neural network module, termed **OrigamiNet**, that can augment any CTC-trained, fully convolutional single line text recognizer, to convert it into a **<u>multi-line version</u>** by providing the model with enough spatial capacity to be able to properly collapse a 2D input signal into 1D without losing information. 
+ *Such modified networks can be trained using exactly their same simple original procedure, and using only unsegmented  image and text pairs.* 
+ We carry out a set of interpretability experiments that show that our trained models **<u>learn an accurate implicit line segmentation</u>**.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209251544948.png)

#### Process



### <u>Primitive Representation Learning for Scene Text Recognition</u>

#### Abstract

*Scene text recognition is a challenging task due to diverse variations of text instances in natural scene images. Conventional methods based on CNN-RNN-CTC or encoder-decoder with attention mechanism may not fully investigate stable and efficient feature representations for multi-oriented scene texts. In this paper, we propose a primitive representation learning method that aims to exploit intrinsic representations of scene text images. We model elements in feature maps as the nodes of an undirected graph. A pooling aggregator and a weighted aggregator are proposed to learn primitive representations, which are transformed into high-level visual text representations by graph convolutional networks. A Primitive REpresentation learning Network (PREN) is constructed to use the visual text representations for parallel decoding. Furthermore, by integrating visual text representations into an encoder- decoder model with the 2D attention mechanism, we pro- pose a framework called PREN2D to alleviate the misalign- ment problem in attention-based methods. Experimental re- sults on both English and Chinese scene text recognition tasks demonstrate that PREN keeps a balance between ac- curacy and efficiency, while PREN2D achieves state-of-the- art performance.*

#### Idea

Old method:	CNN-RNN-CTC or encoder-decoder with attention mechanism

Old method flaws:	May not fully investigate stable and efficient feature representations for multi-oriented scene texts

**<u>*New method:	We propose a primitive representation learning method that aims to exploit intrinsic representations of scene text images.*</u>**

+ We model elements in feature maps as the nodes of an undirected graph. 
+ A pooling aggregator and a weighted aggregator are proposed to learn primitive representations, which are transformed into high-level visual text representations by graph convolutional networks. 
+ A Primitive REpresentation learning Network (PREN) is constructed to use the visual text representations for parallel decoding. 
+ Furthermore, by integrating visual text representations into an encoder- decoder model with the 2D attention mechanism, we propose a framework called PREN2D to alleviate the misalignment problem in attention-based methods. 
+ Experimental results on both English and Chinese scene text recognition tasks demonstrate that PREN keeps a balance between accuracy and efficiency, while PREN2D achieves state-of-the- art performance.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209251641591.png)

#### Process



### <u>Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition</u>

#### Abstract

*Linguistic knowledge is of great benefit to scene text recognition. However, how to effectively model linguistic rules in end-to-end deep networks remains a research challenge. In this paper, we argue that the limited capacity of language models comes from: 1) implicitly language modeling; 2) unidirectional feature representation; and 3) language model with noise input. Correspondingly, we pro- pose an autonomous, bidirectional and iterative ABINet for scene text recognition. Firstly, the autonomous suggests to block gradient flow between vision and language models to enforce explicitly language modeling. Secondly, a novel bidirectional cloze network (BCN) as the language model is proposed based on bidirectional feature representation.Thirdly, we propose an execution manner of iterative correc- tion for language model which can effectively alleviate the impact of noise input. Additionally, based on the ensemble of iterative predictions, we propose a self-training method which can learn from unlabeled images effectively. Extensive experiments indicate that ABINet has superiority on low- quality images and achieves state-of-the-art results on sev- eral mainstream benchmarks. Besides, the ABINet trained with ensemble self-training shows promising improvement in realizing human-level recognition.*

#### Idea

Old method:	Linguistic knowledge is of great benefit to scene text recognition.However, how to effectively model linguistic rules in end-to-end deep networks remains a research challenge.

Old method flaws:	

+ Implicitly language modeling; 
+ Unidirectional feature representation; 
+ Language model with noise input.

**<u>*New method:*</u>**	 

+ **<u>*We propose an autonomous, bidirectional and iterative ABINet for scene text recognition.*</u>** 
+ Firstly, the **<u>*autonomous suggests to block gradient flow between vision and language models*</u>** to enforce explicitly language modeling. 
+ Secondly, a novel **<u>*bidirectional cloze network (BCN) as the language model is proposed based on bidirectional feature representation*</u>**.
+ Thirdly, we propose an **<u>*execution manner of iterative correction for language model which can effectively alleviate the impact of noise input*</u>**. 
+ Additionally, based on the ensemble of iterative predictions, we propose a self-training method which can **<u>*learn from unlabeled images*</u>** effectively.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209251728592.png)

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209251728615.png)
![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209251728616.png)

#### Process



### <u>SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition</u>

#### Abstract

*Scene text recognition is a hot research topic in computer vision. Recently, many recognition methods based on the encoder-decoder framework have been proposed, and they can handle scene texts of perspective distortion and curve shape. Nevertheless, they still face lots of challenges like image blur, uneven illumination, and incomplete char- acters. We argue that most encoder-decoder methods are based on local visual features without explicit global semantic information. In this work, we propose a semantics enhanced encoder-decoder framework to robustly recognize low-quality scene texts. The semantic information is used both in the encoder module for supervision and in the de- coder module for initializing. In particular, the state-of-the- art ASTER method is integrated into the proposed frame- work as an exemplar. Extensive experiments demonstrate that the proposed framework is more robust for low-quality text images, and achieves state-of-the-art results on several benchmark datasets.*

#### Idea

Old method:	Many recognition methods based on the encoder-decoder framework have been proposed, and they can handle scene texts of perspective distortion and curve shape. 

Old method flaws:	Image blur, uneven illumination, and incomplete characters

**<u>*New method:*</u>**	

+ **<u>*We propose a semantics enhanced encoder-decoder framework to robustly recognize low-quality scene texts*</u>**. 
+ The **<u>*semantic information*</u>** is used both in the encoder module for supervision and in the decoder module for initializing. 
+ In particular, the state-of-the- art **<u>*ASTER method*</u>** is integrated into the proposed framework as an exemplar.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209251911766.png)

#### Process



### <u>What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels</u>

#### Abstract

*Scene text recognition (STR) task has a common practice: All state-of-the-art STR models are trained on large synthetic data. In contrast to this practice, training STR models only on fewer real labels (STR with fewer labels) is important when we have to train STR models without synthetic data: for handwritten or artistic texts that are difficult to generate synthetically and for languages other than English for which we do not always have synthetic data. However, there has been implicit common knowledge that training STR models on real data is nearly impossible because real data is insufficient. We consider that this common knowledge has obstructed the study of STR with fewer labels. In this work, we would like to reactivate STR with fewer labels by disproving the common knowledge. We con- solidate recently accumulated public real data and show that we can train STR models satisfactorily only with real labeled data. Subsequently, we find simple data augmen- tation to fully exploit real data. Furthermore, we improve the models by collecting unlabeled data and introducing semi- and self-supervised methods. As a result, we obtain a competitive model to state-of-the-art methods. To the best of our knowledge, this is the first study that 1) shows sufficient performance by only using real labels and 2) in- troduces semi- and self-supervised methods into STR with fewer labels.*

#### Idea

Old method:	All state-of-the-art STR models are trained on large synthetic data.

Old method flaws:	For handwritten or artistic texts that are difficult to generate synthetically and for languages other than English for which we do not always have synthetic data.

**<u>*New method:*</u>**	

+ We consolidate recently accumulated public real data and show that we can train STR models satisfactorily only with real labeled data. 
+ Subsequently, we find simple **<u>*data augmentation to fully exploit real data*</u>**. 
+ Furthermore, we **<u>*improve the models by collecting unlabeled data and introducing semi- and self-supervised methods*</u>**.
+ To the best of our knowledge, this is the first study that 
  + 1) *<u>**Shows sufficient performance by only using real labels**</u>*
    2) *<u>**Introduces semi- and self-supervised methods into STR with fewer labels.**</u>*

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209261549797.png)

#### Process





### <u>Towards Accurate Scene Text Recognition with Semantic Reasoning Networks</u>

#### Abstract

*Scene text image contains two levels of contents: visual texture and semantic information. Although the previous scene text recognition methods have made great progress over the past few years, the research on mining seman- tic information to assist text recognition attracts less at- tention, only RNN-like structures are explored to implic- itly model semantic information. However, we observe that RNN based methods have some obvious shortcomings, such as time-dependent decoding manner and one-way se- rial transmission of semantic context, which greatly limit the help of semantic information and the computation effi- ciency. To mitigate these limitations, we propose a novel end-to-end trainable framework named semantic reasoning network (SRN) for accurate scene text recognition, where a global semantic reasoning module (GSRM) is introduced to capture global semantic context through multi-way parallel transmission. The state-of-the-art results on 7 public bench- marks, including regular text, irregular text and non-Latin long text, verify the effectiveness and robustness of the pro- posed method. In addition, the speed of SRN has significant advantages over the RNN based methods, demonstrating its value in practical use.*

#### Idea

Old method:	Previous scene text recognition methods have made great progress over the past few years, the research on mining semantic information to assist text recognition attracts less attention.Only RNN-like structures are explored to implicitly model semantic information.

Old method flaws:	Time-dependent decoding manner and one-way serial transmission of semantic context, which greatly limit the help of semantic information and the computation efficiency.

**<u>*New method:*</u>**	

+ We propose a novel end-to-end trainable framework named semantic reasoning network (SRN) for accurate scene text recognition, where a **<u>*global semantic reasoning module*</u>** (GSRM) is introduced to **<u>*capture global semantic context through multi-way parallel transmission*</u>**. 
+ The state-of-the-art results on 7 public bench- marks, including regular text, irregular text and non-Latin long text, verify the effectiveness and robustness of the proposed method. In addition, the speed of SRN has significant advantages over the RNN based methods, demonstrating its value in practical use.(Just so good!)

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209261606652.png)

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





### <u>SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition</u>

#### Abstract

*End-to-end scene text spotting has attracted great attention in recent years due to the success of excavating the intrinsic synergy of the scene text detection and recognition. However, recent state-of-the-art methods usually incorporate detection and recognition simply by sharing the backbone, which does not directly take advantage of the feature interaction between the two tasks. In this paper, we propose a new end-to-end scene text spotting framework termed SwinTextSpotter. Using a trans- former encoder with dynamic head as the detector, we unify the two tasks with a novel Recognition Conversion mechanism to explicitly guide text localization through recognition loss. The straightforward design results in a concise framework that requires neither additional rec- tification module nor character-level annotation for the arbitrarily-shaped text. Qualitative and quantitative experi- ments on multi-oriented datasets RoIC13 and ICDAR 2015, arbitrarily-shaped datasets Total-Text and CTW1500, and multi-lingual datasets ReCTS (Chinese) and VinText (Viet- namese) demonstrate SwinTextSpotter significantly outper- forms existing methods.*

#### Idea

Old method:	Excavating the intrinsic synergy of the scene text detection and recognition

Old method flaws:	Recent state-of-the-art methods usually incorporate detection and recognition simply by sharing the backbone, which does not directly take advantage of the feature interaction between the two tasks.

**<u>*New method:	We propose a new end-to-end scene text spotting framework termed SwinTextSpotter.*</u>** 

+ Using a **<u>*transformer encoder with dynamic head as the detector*</u>**, we unify the two tasks with a novel Recognition Conversion mechanism to explicitly **<u>*guide text localization through recognition loss*</u>**. 
+ The straightforward design results in a concise framework that requires neither additional rectification module nor character-level annotation for the arbitrarily-shaped text.

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209252012026.png)

#### Process



### <u>Text Spotting Transformers</u>

#### Abstract

*In this paper, we present TExt Spotting TRansformers (TESTR), a generic end-to-end text spotting framework using Transformers for text detection and recognition in the wild. TESTR builds upon a single encoder and dual decoders for the joint text-box control point regression and character recognition. Other than most existing litera- ture, our method is free from Region-of-Interest operations and heuristics-driven post-processing procedures; TESTR is particularly effective when dealing with curved text-boxes where special cares are needed for the adaptation of the tra- ditional bounding-box representations. We show our canon- ical representation of control points suitable for text in- stances in both Bezier curve and polygon annotations. In addition, we design a bounding-box guided polygon de- tection (box-to-polygon) process. Experiments on curved and arbitrarily shaped datasets demonstrate state-of-the- art performances of the proposed TESTR algorithm.*

#### Idea

Old method:	None

Old method flaws:	None

**<u>*New method:*</u>**	

+ **<u>*TESTR builds upon a single encoder and dual decoders for the joint text-box control point regression and character recognition.*</u>** 
+ Other than most existing literature, our method is free from Region-of-Interest operations and heuristics-driven post-processing procedures; 
+ TESTR is particularly effective when dealing with **<u>*curved text-boxes*</u>** where special cares are needed for the adaptation of the traditional bounding-box representations. 
+ We show our **<u>*canonical representation of control points*</u>** suitable for text instances in both **<u>*Bezier curve and polygon annotations*</u>**. 
+ In addition, we design a **<u>*bounding-box guided polygon detection*</u>** (box-to-polygon) process. 

#### Figure

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209261556239.png)

![](https://raw.githubusercontent.com/jianlai2600/IMAGE/main/img/202209261559092.png)

#### Process



























