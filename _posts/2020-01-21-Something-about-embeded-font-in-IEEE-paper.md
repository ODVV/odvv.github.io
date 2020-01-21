---
layout: post
title:  "投稿时的PDF检测中出现问题的解决方法"
date:   2020-01-21
excerpt: "页边距啦，字体嵌入啦，图像格式啦"
tag:
- 论文投稿
- LeTex
- PDF检测
---

一般IEEE会议论文进行投稿提交之前都要进行 PDF 的检测，我在提交的时候主要检查了两项内容：
## 检查页边距是否符合规定
   
   基本按照模板来写文章，不更改模板里的设置，这一项基本不会有问题。
   
   ![](/images/posts/2020-01-21-temple-margin.jpg)
## 检查是否包含非嵌入的字体
   
   在使用 LaTex 写文章插入图片时，所使用的图片格式为 *.pdf, *.eps 图中包含了多种字体，遇到了以下两种问题：

### 字体相关：Fonts is not embedded，字体嵌入存在问题
   
   在 PDF 应当使用嵌入的字体而不是系统字体，有两种解决方案：

#### office 相关的解决方案
  
在使用 office 软件将相关文档转存为 pdf 格式时，在另存为弹出窗口时选择“选项”，勾选【PDF选项】下的“符合 ISO 19005-1 标准(PDF/A)”，确定后进行下一步。

![](/images/posts/2020-01-21-word-solution.png)

#### Adobe Acrobat 相关解决方案

使用 Adobe Acrobat 将不标准的 PDF 文档另存为标准的符合 IEEE 所要求的格式。

* 首先需要下载对应的 joboption 设置文件，下载地址：

    [Acrobat 5](http://controls.papercept.net/conferences/support/files/Acro5.recommended24Oct2006.joboptions.zip), [Acrobat 6](http://controls.papercept.net/conferences/support/files/Acro6.recommended.24Oct2006.joboptions.zip),
    [Acrobat 7](http://controls.papercept.net/conferences/support/files/Acro7.recommended.24Oct2006.joboptions.zip),
    [Acrobat 8](http://controls.papercept.net/conferences/support/files/Acro8.recommended.6Dec2006.joboptions.zip),
    [Acrobat 9 及之后的版本](http://controls.papercept.net/conferences/support/files/Acro9.recommended.9Dec2008.joboptions.zip)

    下载完成后解压得到 .joboption 的文件，之后需要使用 Adobe Acrobat Distiller 来添加设置。

    打开 Adobe Acrobat Distiller 后点击 “设置”，“添加 Adobe PDF 设置”，到 .joboption 的文件所在路径完成添加。

* 接着使用 Adobe Acrobat 将待处理 PDF， 通过“打印”，【打印机】选择"Adobe PDF",点击属性。

    在接下来选择“Adobe PDF 设置”选项卡，将默认设置选择上一步所安装的设置，进行确定，进一步确定和保存。

    ![](/images/posts/2020-01-21-adobe-solution.png)


### Hit Error Missing XObject

首先看一下官方给的解释：

> Internal pdf error, caused by one of the figures. This seems to occur when figures are directly saved as pdf from Matlab or Xfig. Try to resolve this by rasterizing the figures (converting to jpeg or png). Avoid progressive jpeg. Some users could work around this issue by converting all figures to eps.

> Sometimes re-distilling the pdf makes the problem go away: Open the pdf in Adobe Acrobat Pro and save to postcript. Use Adobe Distiller to recompile the pdf.

> If all fails send us the pdf for inspection.

大意就是当把一张图直接从 Matlab 或者 Xfig 直接保存为 pdf，在文档里插入了这样图片的 PDF 时会产生这样的错误。

解决的方法包括：

* 把图栅格化（转存为jpeg或png再用）

* 有时重新提取 pdf 可以解决问题：在Adobe Acrobat Pro中打开pdf并保存到后抄本。使用Adobe Distiller重新编译pdf。