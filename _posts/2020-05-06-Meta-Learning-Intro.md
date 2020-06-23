---
layout: post
title:  "元学习：Meta learning"
date:   2020-05-06
excerpt: "学会学习：Learn to learn"
tag:
- 元学习
- 笔记
---

## 元学习的期望

机器在学习了过往的任务的基础上，可以成为一个更好的学习者。在面对新的任务时，可以学习得更快，因为在过往的任务中学习到了学习的技巧。

## 与 life-long learning 的区别

life-long learning 用一个模型处理所有模型，meta learning 对应每个任务有一个模型。

## 与一般机器学习的区别

![](/images/posts/2020-05-06-meta-learning1.png)

一般机器学习问题：给数据，学习*f*，处理特定任务，输出网络参数。

元学习问题：给数据，学习*F*，对应不同任务，学习对应*f*，处理特定任务，输出对应*f*。

**直接的理解是，Meta learning 学习不同的任务，学习到对应每个任务给出任务的处理函数（对应分类任务就是分类器）的能力——也称为一个学习算法，从而在新的任务上，该学习算法可以给出对应新任务的任务处理函数。**

## Meta learning 的训练和验证

![](/images/posts/2020-05-06-meta-learning2.png)

## 不同的 Meta learning

![](/images/posts/2020-05-06-meta-learning3.png)