---
layout: post
title:  "一种无监督的分类器集成方法"
date:   2019-05-29
excerpt: "对于全新的无标签数据，如何利用与新数据独立同分布的数据训练的现有分类器来完成对新数据的有效分类？"
tag:
- spectral analysis
- classifier balanced accuracy
- unsupervised learning
- Reading Notes
---

## 文献清单
[1] Parisi F , Strino F , Nadler B , et al. Ranking and combining multiple predictors without labeled data[J]. Proceedings of the National Academy of Sciences, 2014, 111(4):1253-1258.

这是一篇比较老的文献了，2012年投稿，2014年见刊，发表在PNAS。作者 [Fabio Parisi](https://www.linkedin.com/in/wwxww) 来自耶鲁大学医学院病理学系，通讯作者是 [Yuval Kluger](https://medicine.yale.edu/bbs/people/yuval_kluger.profile)。

## 内容简介

考虑分类问题中的这样的应用场景，已有一系列的未知分类性能的分类器，对于新的无标签数据如何用这些已有的分类器完成分类。

这一分类问题与标准的有监督分类问题有很大差异，在新的数据有一些标签的情况下，我们可以来估计每个已有分类器的性能。

文章主要解决的问题就是在新数据无标签的情况下，如何对每个现有分类器的可靠性进行评估以及对多个分类器进行集成，使得新数据能够获得更好的分类结果。作者提出一种谱方法来解决这两个问题。

## 问题提出
考虑二分类的问题，类标签为 $\{+1,-1\}$ 分别对应目标类(Positive)和非目标类(Negative)。

已有的 $M$ 个未知性能的分类器为 $\{f_i\}_{i=1}^M$，未标记的数据 $D=\{x_k\}_{k=1}^S$，其中 $S$ 表示每个未标记样本的实例。对应这些样本的真实的类标签（未知）为 $\boldsymbol{y}=(y_1,\dots,y_s)$。

我们将每个样本实例及其标签表示成一个随机向量 $(X,Y)\in \ \times\{+1,-1\}$ 对应概率密度函数为 $p(x,y)$ ，边界分布为 $p_X(x), p_Y(y)$。
性能评价标准：平衡精度 $\pi$

$\pi=\frac{sensitivity + specificity}{2}=\frac{1}{2}(\psi+\eta)$

### 前提假设
1. 每个分类器都是由不同的数据训练得到的。
2. 已有的各个分类器之间条件独立
3. $S$ 中的样本相互独立并具有相同的边缘分布

### 分类器排序
作者提出的谱方法来实现在数据无标签的情况下对 $M$ 个分类器性能进行排序的方法，首先简述该过程的步骤如下：

* 将测试数据集 $D$ 分别作为 $M$ 个分类器的输出
* 得到 $M$ 个分类结果，将这些分类结果组成 $M\times S$ 矩阵 $T$
* 对 $T$ 计算样本协方差矩阵 $Q^{M\times M}$
* 对 $Q$ 矩阵进行特征分解，得到对应特征值最大的特征向量中各个分量的大小作为对应分类器的性能排序的指标

-------------------
## To Be Continue

矩阵的每项如下： 
$q_{ij}=E[(f_i(X)-\mu_i)(f_j(X)-\mu_j)]$

**推论**

该推论说明 $Q$ 矩阵的非对角线项与一个秩为1的矩阵 $R=\lambda \bm{v}\bm{v}^T$的非对角线项是相同的，矩阵 $R$ 的的特征向量为单位向量，特征值为 $\lambda=(1-b^2)\cdot \sum_{i=1}^M(2\pi-1)^2$
