---
layout: post
title:  "拍扁这些不确定数量的变量"
date:   2019-06-03
excerpt: "在可迭代对象或者函数的输入中，往往存在多个元素或者变量的情况，所以如何处理这些可变的变量呢？"
tag:
- Python
- 可迭代对象
- 函数
---

> 最开始接触 Python 是因为在课程学习时要完成编程作业，老师要求用 Python。之后从事逻辑开发的方向，就再没接触它了。而现在脱离硬件之后，又要重拾 Python 了（<del>菜鸡</del>），现在只会面向 google 编程。这里也记录一下自己重新学习的一些 Python 的知识和技巧。

本篇文章相关内容，来自 <code>Python3 Cookbook</code> 3rd Edition，译者的[yidao620c](https://github.com/yidao620c/python3-cookbook/blob/master/source/index.rst)。

------------------------
## 解压可迭代对象赋值给多个变量

### 问题
如果一个可迭代对象的元素个数超过变量个数时，会抛出一个 ValueError 。 那么怎样才能从这个可迭代对象中解压出 N 个元素出来？

### 解决方案
Python 的星号表达式可以用来解决这个问题。比如，你在学习一门课程，在学期末的时候， 你想统计下家庭作业的平均成绩，但是排除掉第一个和最后一个分数。如果只有四个分数，你可能就直接去简单的手动赋值， 但如果有 24 个呢？这时候星号表达式就派上用场了：

```python
def drop_first_last(grades):
    first, *middle, last = grades
    return avg(middle)
```

另外一种情况，假设你现在有一些用户的记录列表，每条记录包含一个名字、邮件，接着就是不确定数量的电话号码。 你可以像下面这样分解这些记录：

```python
>>> record = ('Dave', 'dave@example.com', '773-555-1212', '847-555-1212')
>>> name, email, *phone_numbers = record
>>> name
'Dave'
>>> email
'dave@example.com'
>>> phone_numbers
['773-555-1212', '847-555-1212']
>>>
```
值得注意的是上面解压出的 phone_numbers 变量永远都是列表类型，不管解压的电话号码数量是多少（包括 0 个）。 所以，任何使用到 phone_numbers 变量的代码就不需要做多余的类型检查去确认它是否是列表类型了。
星号表达式也能用在列表的开始部分。比如，你有一个公司前 8 个月销售数据的序列， 但是你想看下最近一个月数据和前面 7 个月的平均值的对比。你可以这样做：
*trailing_qtrs, current_qtr = sales_record
trailing_avg = sum(trailing_qtrs) / len(trailing_qtrs)
return avg_comparison(trailing_avg, current_qtr)
下面是在 Python 解释器中执行的结果：

```python
>>> *trailing, current = [10, 8, 7, 1, 9, 5, 10, 3]
>>> trailing
[10, 8, 7, 1, 9, 5, 10]
>>> current
3
```
-----------------------

## 可接受任意数量参数的函数
###问题
你想构造一个可接受任意数量参数的函数。
解决方案
为了能让一个函数接受任意数量的位置参数，可以使用一个*参数。例如：

```python
def avg(first, *rest):
    return (first + sum(rest)) / (1 + len(rest))
# Sample use
avg(1, 2) # 1.5
avg(1, 2, 3, 4) # 2.5
```

在这个例子中，rest是由所有其他位置参数组成的元组。然后我们在代码中把它当成了一个序列来进行后续的计算。
为了接受任意数量的关键字参数，使用一个以**开头的参数。比如：


```python
import html
def make_element(name, value, **attrs):
    keyvals = [' %s="%s"' % item for item in attrs.items()]
    attr_str = ''.join(keyvals)
    element = '<{name}{attrs}>{value}</{name}>'.format(
                name=name,
                attrs=attr_str,
                value=html.escape(value))
    return element
# Example
# Creates '<item size="large" quantity="6">Albatross</item>'
make_element('item', 'Albatross', size='large', quantity=6)
# Creates '<p>&lt;spam&gt;</p>'
make_element('p', '<spam>')
```

在这里，attrs 是一个包含所有被传入进来的关键字参数的字典。
如果你还希望某个函数能同时接受任意数量的位置参数和关键字参数，可以同时使用*和**。比如：

```python
def anyargs(*args, **kwargs):
    print(args) # A tuple
    print(kwargs) # A dict
```
使用这个函数时，所有位置参数会被放到args元组中，所有关键字参数会被放到字典kwargs中。
讨论
一个\*参数只能出现在函数定义中最后一个位置参数后面，而 \*\*参数只能出现在最后一个参数。 有一点要注意的是，在\*参数后面仍然可以定义其他参数。


```python
def a(x, *args, y):
    pass
def b(x, *args, y, **kwargs):
    pass
```
这种参数就是我们所说的强制关键字参数，在后面7.2小节还会详细讲解到。
