---
layout: post
title:  "[转载] 如何训练稳定的GAN"
date:   2019-09-18
excerpt: "训练很受罪"
tag:
- GAN
- 转载
- 网络训练
---

> 如何训练稳定的生成对抗网络 [原文地址](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)


生成性对抗网络，简称 GANs ，是一种利用深卷积神经网络等深度学习方法进行生成性建模的方法。

尽管由 GANs 生成的结果是显著的，但是训练一个稳定的模型是一个挑战。原因是训练过程本身不稳定，导致两种竞争模型同时进行动态训练。

尽管如此，由于大量的经验试验和误差，但是许多实践者和研究者发现并报道了少量的模型结构和训练配置，从而导致了稳定 GAN 模型的可靠训练。

在这篇文章中，您将发现稳定的一般对抗性网络模型的配置和训练的经验启发。



读完这篇文章，你会知道：

* GANs 中生成器和鉴别器模型的同时训练具有内在的不稳定性。

* 经验性地发现的 DCGAN 结构为大多数 GAN 应用提供了一个可靠的起点。

* GANs 的稳定训练仍然是一个开放的问题，许多其他经验发现的技巧和技巧已经被提出并可以立即采用。


## 概述

本教程分为三个部分，分别是：

1. 训练 GAN 的挑战
2. 深卷积 GAN
3. 其他提示和技巧

## 生成性对抗网络训练的挑战

生成对抗网络很难训练。它们难以训练的原因是在游戏中同时训练了生成器模型和鉴别器模型。这意味着对一个模型的改进是以牺牲另一个模型为代价的。

训练两个模型的目标是在两个相互竞争的关注点之间找到一个平衡点。

> 训练 GANs 的关键在于找到两人非合作博弈的纳什均衡。[…]不幸的是，找到纳什均衡是一个非常困难的问题。算法存在于特殊情况下，但我们不知道任何适用于 GAN 博弈的可行算法，其中代价函数是非凸的，参数是连续的，参数空间是非常高维的。
> — [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), 2016.


这也意味着，每次更新其中一个模型的参数时，正在求解的优化问题的性质都会发生变化。
这会产生创建动态系统的效果。

> 但是对于GAN，就像下山的每一步的过程中都一点点改变了整个景观。这是一个动态系统，优化过程不是寻求最小值，而是寻求两种力之间的平衡。
> — Page 306, [Deep Learning with Python](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/ref=as_li_ss_tl?keywords=deep+learning&qid=1553730173&s=books&sr=1-2&linkCode=sl1&tag=inspiredalgor-20&linkId=680904778b3cf1d6c567f3ff3ec4b48b&language=en_US), 2017.

就神经网络而言，同时训练两个相互竞争的神经网络的技术挑战是它们可能无法收敛。

> 研究人员应该努力解决的最大问题是不收敛问题。
>  — NIPS 2016 [Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160), 2016.

相比于不收敛，GANs 可能遇到其他少数几种失效模式。

> 一种常见的失效模式是，生成器在生成域中的特定示例之间振荡，而不是找到平衡点。
> 在实践中，GANs 经常看起来是振荡的，[…]意味着它们从生成一种样本到生成另一种样本，而最终没有达到平衡。
> — NIPS 2016 [Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160), 2016.

可能最具挑战性的模型故障是对于生成器的多个输入生成了相同输出的情况。

这被称为“模式崩溃”（Mode collapse），可能是训练网络中最具挑战性的问题之一。

> 模式崩溃，也就是，是当生成器学会将多个不同的输入z值映射到同一输出点时发生的问题。
> — NIPS 2016 [Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160), 2016.

最后，没有一个好的客观指标来评估 GAN 在训练期间是否表现良好。看loss是不够的。
相反，最好的方法是目视检查生成的示例并使用主观评估。

> 生成性对抗网络缺乏目标函数，难以比较不同模型的性能。一个直观的性能度量可以通过让人类注释器判断样本的视觉质量来获得。
> — [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), 2016.

在撰写本文的时候，关于如何设计和训练 GAN 模型还没有很好的理论基础，但是有一个已经建立的启发式的文献，或者说“hacks”，已经在实践中被证明是有效的。

## 深层卷积生成对抗网络

在设计和训练稳定的 GAN 模型方面，最重要的一步也许是alec radford等人2015年的论文。题为“具有深层卷积生成对抗网络的无监督表征学习”。

在文中，他们描述了深卷积GAN，或 DCGAN，已经成为 GAN 发展的事实标准。

>  GAN 学习的稳定性仍然是一个有待解决的问题。幸运的是，当仔细选择模型结构和超参数时， GAN 学习表现良好。Radford等人（2015）制作了一种深卷积 GAN （dcgan），它在图像合成任务中表现非常好……
> — Page 701, Deep Learning, 2016.

![](/images/posts/Example-of-the-Generator-Model-Architecture-for-the-DCGAN-1024x440.png)

文中的研究成果是辛苦获得的，是在经过大量的经验试验和错误之后，在不同的模型架构、配置和训练方案下得出的。在开发新的 GANs （至少对于基于图像合成的任务而言）时，他们的方法仍然被强烈推荐为一个起点。

> ……经过广泛的模型探索，我们确定了一系列体系结构，这些体系结构在一系列数据集上实现了稳定的训练，并允许训练更高分辨率和更深层次的生成模型。
> — Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2015

以下是本文对 GAN 架构建议的总结:

![](/images/posts/Summary-of-Architectural-Guidelines-for-Training-Stable-Deep-Convolutional-Generative-Adversarial-Networks-1024x279.png)

一下进行逐一解释：

### Use Strided Convolutions

在卷积神经网络中，通常使用诸如max pooling layers这样的池化层来进行下采样。
在 GANs 中，建议不要使用池化层，而是使用卷积层中的stride在鉴别器模型中执行下采样。
类似地，分数步长（反卷积层）可用于发生器中的上采样。

> [替换]确定性空间池函数（如max pooling）使用跨步卷积，允许网络学习自己的空间下采样。我们在生成器中使用这种方法，允许它学习自己的空间上采样和鉴别器。

### 移除全连接层

通常在卷积层中的特征提取层之后使用全连接层来解释在模型的输出层之前提取的特征。

相反，在 GANs 中，不使用全连接层，在鉴别器中，卷积层被展平并直接传递到输出层。

另外，传递到生成器模型的随机高斯输入向量被直接重塑为多维张量，该多维张量可以传递到第一卷积层，以准备上尺度。

> 以均匀噪声分布z为输入的 GAN 的第一层可以称为全连通层，因为它只是一个矩阵乘法，但其结果被reshape为四维张量，并用作卷积堆栈的开始。对于鉴别器，最后一个卷积层被展平，然后馈入单个sigmoid输出。

### 使用批量正则化

批量规范化将前一层的激活标准化，使平均值和单位方差为零。这有稳定训练过程的效果。

在训练深卷积神经网络时，批处理规范化已成为一种主要的训练方法，而 GANs 也不例外。在鉴别器和生成器模型中，除了生成器的输出和鉴别器的输入外，建议使用批处理规范层。

> 然而，直接对所有层应用batchnorm会导致样本振荡和模型不稳定。通过不对生成器输出层和鉴别器输入层应用batchnorm，避免了这种情况。

### 使用 ReLU, Leaky ReLU, and Tanh

像relu这样的激活函数被用来解决深卷积神经网络中的消失梯度问题，并促进稀疏激活（例如许多零值）。

建议对生成器使用relu，但对鉴别器模型不使用relu。相反，在鉴别器中优选允许值小于零的relu的变体，称为leakyrelu。

> 除了使用tanh函数的输出层外，relu激活在生成器中使用。[…]在鉴别器中，我们发现漏校正激活工作良好…

此外，生成器在输出层中使用双曲正切（tanh）激活函数，并且对生成器和鉴别器的输入被缩放到范围[-1，1]。

> 除了调整tanh激活函数的范围外，没有对训练图像进行预处理。

模型权值初始化为小高斯随机值，鉴别器中leakyrelu的斜率初始化为0.2。

> 所有权重均由标准差为0.02的零中心正态分布初始化。在leakyrelu中，所有模型的泄漏斜率都设置为0.2。

### 使用 Adam Optimization

生成器和鉴别器都接受随机梯度下降训练，训练的批大小适中，为128幅图像。

> 所有模型均采用小批量随机梯度下降（sgd）训练，最小批量为128。

具体来说，ADAM版本的随机梯度下降被用来训练学习率为0.0002和动量（beta1）为0.5的模型。

> 我们使用带有优化超参数的adam优化器。我们发现建议的学习率为0.001，太高了，改用0.0002。此外，我们发现动量项 beta1 在建议值为 0.9 时会导致训练振荡和不稳定，而将其减小到 0.5 则有助于稳定训练。

## 其他提示和技巧

dcgan论文为配置和训练生成器和鉴别器模型提供了一个很好的起点。

此外，还编写了一些回顾性的演示文稿和论文来总结这些以及用于配置和培训 GANs 的其他启发式方法。

在本节中，我们将介绍其中的一些，并重点介绍一些需要考虑的附加提示和技巧。

Tim Salimans等人于 2016 年发表的论文。在 openai 中，题为“改进的 GANs 训练技术”列出了五种在训练 GANs 时需要考虑的改进收敛性的技术。

* 特征匹配。使用半监督学习开发 GAN 。
* 小批量鉴别。在一个小批量中开发多个样本的特性。
* 历史平均值。更新损失函数以合并历史记录。
* 单面标签平滑。将鉴别器的目标值从1.0缩放。
* 虚拟批处理规范化。使用实际图像的参考批次计算批次范数统计。

伊恩·古德费罗在2016年NIPS会议上的 GANs 指南中详细阐述了其中一些更成功的建议，这些建议写在题为“指南：生成性对抗性网络”的论文中。具体来说，第4节题为“技巧和诀窍”，其中描述了四种技术。

1. 训练中使用标签
在 GANs 中使用标签可以提高图像质量。

> 以任何方式使用标签，形状或形式几乎总是导致模型生成的样本的主观质量的显著提高。

2. 单侧标签平滑
在值为0.9的鉴别器中使用目标作为实例或使用随机范围的目标可以获得更好的结果。

>单侧标签平滑的思想是用一个略小于1的值替换实际例子的目标，例如.9[…]这防止了鉴别器中的极端外推行为…

3. 虚拟批处理规范化
对真实图像或一幅生成图像的真实图像进行批量统计比较好。

> …可以使用虚拟批处理规范化，其中每个示例的规范化统计信息是使用该示例和引用批的并集计算的

4. 能平衡G和D吗？

根据损失的相对变化在生成器或鉴别器中调度或多或少的训练是直观的，但不可靠。

> 实际上，鉴别器通常更深，有时每层的滤波器比生成器多。


## 小结
A summary of some of the more actionable tips is provided below.

* Normalize inputs to the range [-1, 1] and use tanh in the generator output.
* Flip the labels and loss function when training the generator.
* Sample Gaussian random numbers as input to the generator.
* Use mini batches of all real or all fake for calculating batch norm statistics.
* Use Leaky ReLU in the generator and discriminator.
* Use Average pooling and stride for downsampling; use ConvTranspose2D and stride for upsampling.
* Use label smoothing in the discriminator, with small random noise.
* Add random noise to the labels in the discriminator.
* Use DCGAN architecture, unless you have a good reason not to.
* A loss of 0.0 in the discriminator is a failure mode.
* If loss of the generator steadily decreases, it is likely fooling the discriminator with garbage images.
* Use labels if you have them.
* Add noise to inputs to the discriminator and decay the noise over time.
* Use dropout of 50 percent during train and generation.