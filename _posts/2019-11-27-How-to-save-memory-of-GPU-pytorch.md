---
layout: post
title:  "Pytorch 减少显存占用"
date:   2019-09-18
excerpt: "避免 out of Memory，显存爆炸"
tag:
- 深度学习
- Pytorch
- 转载
- 网络训练
---

搜了一下有如下几条，主要来自知乎和CSDN，暂时先放这里，之后再整理。

* 首先想到的当然还是减少batch size 和输入数据的样本数
* 尽可能使用inplace操作， 比如relu 可以使用 inplace=True 
* 每次循环结束时 删除 loss，可以节约很少显存
* 使用float16精度混合计算。NVIDIA英伟达 apex，很好用，可以节约将近50%的显存，但要小心一些不安全的操作如 mean和sum，溢出fp16
* 记录和获取数据时，使用loss += loss.detach()来获取不需要梯度回传的部分
* 每次迭代都会引入点临时变量，会导致训练速度越来越慢，基本呈线性增长。但如果周期性的使用torch.cuda.empty_cache()的话就可以解决这个问题
* 用checkpoint牺牲计算速度：在Pytorch-0.4.0出来了一个新的功能，可以将一个计算过程分成两半，也就是如果一个模型需要占用的显存太大了，我们就可以先计算一半，保存后一半需要的中间结果，然后再计算后一半。新的checkpoint允许我们只存储反向传播所需要的部分内容。如果当中缺少一个输出(为了节省内存而导致的)，checkpoint将会从最近的检查点重新计算中间输出，以便减少内存使用(当然计算时间增加了)
* torch.backends.cudnn.benchmark = True 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销