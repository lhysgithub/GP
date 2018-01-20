2017/12/26 

一切终结结束，一切又会有新的开始。从今天起，我将纵身于学术，一去不复返。

学习TensorFlow官方文档中文版基本使用方法。

最终还是玩儿过去了~

2017/12/27

代码中 assign() 操作是图所描绘的表达式的一部分, 正如 add() 操作一样. 所以在调用 run() 执行表达式之前, 它并不会真正执行赋值操作.

我选择去看TensorFlow英文文档。

通读[GetStarted](https://www.tensorflow.org/get_started/get_started)文档

如果需要更多的编码需求参考[ProgrammerGuide](https://www.tensorflow.org/programmers_guide)文档

本次学习重要心得就是中文文档看不懂一定要去读英文文档，中文的文档无论是逻辑排列还是词汇使用都不适合理解，这或许和我们高中做了三年的阅读理解有关，这样还能顺便学习一下不认识的单词。

2018/01/15

确立了使用遗传算法作为搜索算法

确立了使用通过找到特征代表的方式可解释的选取修改的像素点

确立了在搜索时引入置信因子的思想

2018/01/20

[Keras:基于Python的深度学习库](https://keras-cn.readthedocs.io/en/latest/)

[匿名函数](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/0013868198760391f49337a8bd847978adea85f0a535591000)

[生成器](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/00138681965108490cb4c13182e472f8d87830f13be6e88000)

[正则化、经验风险、结构风险](https://www.zhihu.com/question/20924039)

[TRACEBACK](http://blog.csdn.net/handsomekang/article/details/9373035)

[Python_Try](http://www.runoob.com/python/python-exceptions.html)

提出疑问：

1、一个5X5卷积核卷积操作32X32X3的图片，得到的输出究竟是28X28X3的图像还是28X28X1的图像？
我认为是28X28X3否则会丢失颜色比例信息，当然合成一个值有的时候也可以保留比例信息，但是肯定会丢失一些信息！（具体如何做到要看keras源码）

2、一个28X28X1的图像经过32个3X3卷积核，得到32个26X26X1的图像，那么在经过32个3X3卷积核是如何再次得到32个24X24X1的图像的？（具体如何做到要看keras源码）

![MINIST针对LeNet网络架构的一副图片的前向传导过程](https://images0.cnblogs.com/blog/571288/201311/26095358-97bf2e756b3248ef838904916b0997ea.png)

上图为：MINIST针对LeNet网络架构的一副图片的前向传导过程

针对卷积层工作原理进行如下参考文献学习：

[利用卷积神经网络识别CIFAR-10](https://limengweb.wordpress.com/2016/12/31/%E5%88%A9%E7%94%A8%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%86%E5%88%ABcifar-10/)

[[CNN中各层图像大小的计算](http://blog.csdn.net/gavin__zhou/article/details/50609325)](http://blog.csdn.net/gavin__zhou/article/details/50609325)