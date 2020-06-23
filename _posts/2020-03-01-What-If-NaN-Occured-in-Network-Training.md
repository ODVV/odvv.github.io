---
layout: post
title:  "如何处理网络训练中产生NaN的问题"
date:   2020-03-01
excerpt: "在网络训练中有时会产生输出结果包含NaN的情况，导致这种错误的原因可能是多方面的。"
tag:
- 网络训练
- 梯度爆炸
- 笔记
---

一种 NaN 导致的错误：

```python
RuntimeError: cudaEventSynchronize in future::wait: device-side assert triggered
```

这个错误是由于在使用 BCELoss 两分类交叉熵损失时，预测标签的输入需要在 0 - 1 范围之内，而 NaN 的出现则产生改错误。

在网络训练中出现 NaN 的可能原因入下：
* 梯度爆炸
  * loss随着每轮迭代越来越大，最终超过了浮点型表示的范围，就变成了NaN
  * 一个有效的方式是增加“gradient clipping“
* 学习率过大
  * 降低 Batch_Size
  * 降低学习率
* 网络结构不稳定
  * 增加BN
  * 检查网络中是否 0 作为了除数或者 0 或负数计算自然对数
* 数据问题
  * 坏数据，比如数据输入中就包含 NaN
  * 对数据进行归一化