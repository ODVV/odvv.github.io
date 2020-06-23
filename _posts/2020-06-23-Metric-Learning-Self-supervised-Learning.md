---
layout: post
title:  "度量学习：笔记 2"
date:   2020-06-23
excerpt: "度量学习 自监督学习"
tag:
- 度量学习
- 笔记
---

## 从表征学习开始

深度学习的成果在于CNN的强大的特征提取能力。

表征十分重要：如何学习好的表征？

学习可分为有监督和无监督。有监督的过程是依据每个图像的标签，进行端到端的训练得到特征提取。无监督学习不需要显式的标签，构造无需标注的监督信息，时而也称为自监督学习。

自监督学习逐渐称为重要的方向，由于真实世界中绝大多数都是无标签的数据。

## 直接建模分布 p(x)

### 自编码
![](/images/posts/20200623-15.png)

### GAN
![](/images/posts/20200623-16.png)


## Pretext 自监督学习的前置任务

![](/images/posts/20200623-17.png)

### 一些可用的pretext

![](/images/posts/20200623-18.png)

![](/images/posts/20200623-19.png)

![](/images/posts/20200623-20.png)

### 基于图像patch的pretext的局限性

![](/images/posts/20200623-21.png)

随着网络层的增加到一定程度，所提取的特征，用于分类的性能会降低。

### CPC
![](/images/posts/20200623-22.png)

### Instance Discrimination
![](/images/posts/20200623-23.png)

把每个样本当做一个类别。


### Constractive Multi-view Coding - CMC
![](/images/posts/20200623-24.png)

![](/images/posts/20200623-25.png)

![](/images/posts/20200623-26.png)

![](/images/posts/20200623-27.png)

### Metric Learning 视角
![](/images/posts/20200623-28.png)


[视频课程](https://www.bilibili.com/video/BV1Qt4y117YS)