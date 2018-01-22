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

[Keras Documentation](https://keras.io/)

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

对应以上两个问题，我查看了conv2d附近的源码（构建计算图的部分），并未找到答案，看来答案可能在运行计算图的部分。

针对卷积层工作原理进行如下参考文献学习：

[利用卷积神经网络识别CIFAR-10](https://limengweb.wordpress.com/2016/12/31/%E5%88%A9%E7%94%A8%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%86%E5%88%ABcifar-10/)

[[CNN中各层图像大小的计算](http://blog.csdn.net/gavin__zhou/article/details/50609325)](http://blog.csdn.net/gavin__zhou/article/details/50609325)

[图像分割-知乎](https://www.zhihu.com/question/51567094)

the rank of the convolution：卷积的阶，是指卷积核的阶。2阶就是图像的

卷积核是对应像素点的，像素点通过XY坐标来确定，所以卷积核应该用2阶的。

[Python继承](http://www.cnblogs.com/Joans/archive/2012/11/09/2757368.html)

github desktop 上传commit时所用账户和git选项中email有关！

待看：

[机器学习算法实践——K-Means算法与图像分割](http://blog.csdn.net/google19890102/article/details/52911835)

[OpenCV-Python-Tutorial](https://github.com/makelove/OpenCV-Python-Tutorial/tree/master/ch27-%E5%88%86%E6%B0%B4%E5%B2%AD%E7%AE%97%E6%B3%95%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2)

[Pycharm快捷键](http://blog.csdn.net/pipisorry/article/details/39909057)

[StackoverflowAboutPython](https://taizilongxu.gitbooks.io/stackoverflow-about-python/content/index.html)

多重继承时：查找顺序（MRO）、重复调用（钻石继承）问题



2018/01/22

计划：

1. 完成基础遗传算法的实现
2. 完成一片托福试卷的练习

实行：

1. [Markdown文档](http://wowubuntu.com/markdown/#p)
2. [numpy.argmax](http://www.cnblogs.com/zhouyang209117/p/6512302.html)
3. [MINIST数据集格式](https://www.jianshu.com/p/84f72791806f)
4. [tf.reduce_sum](https://www.zhihu.com/question/51325408)
5. [使用转换mnist数据库保存为bmp图片]http://blog.csdn.net/u010194274/article/details/50817999)
6. [MNIST数据集转换为图像](http://blog.csdn.net/u012507022/article/details/51376626)
7. [使用 tfdbg 调试 TensorFlow 模型](http://developers.googleblog.cn/2017/03/tfdbg-tensorflow.html)
8. ​


提出问题:

1. L2攻击中`self.tlab`内容存的是什么?(图片数量,分类种类),为什么目标分类还会有所有分类种类的概率向量,默认为											                       $(P_1,P_2,..,P_t,...,P_{10}),其中t为目标分类,P_t=1;P_i=0,i \ne t ?$




学习心得:
1. ​
```python
#np.argmax(data.test_labels[start+i])
#说明data.test_labels[start+i]每一个单项都是一个概率向量
```
2. tensorflow因为引入了无限维度的张量,所以在计算的过程中产生了计算方向的问题,或者说实在哪个维度上进行计算,在第0维度上进行计算,就是对最外层进行计算.例如:

   [[1,2,3],[4,5,6]]在第0维度上操作就是[[5,7,9]]如果不保留多余维度将得到[5,7,9]

   [[1,2,3],[4,5,6]]在第1维度上操作就是[[6],[15]]如果不保留多余维度将得到[6,15]

   例如在第K维度上操作,那么在第K维度上对应项相操作.

3. 张量中shape用来存的是每一维内容的数量.

   例如[[1,2,3],[4,5,6]]表示为张量的shape形式为(2,3)

4. MINIST数据集数据格式:

   文件的格式很简单，可以理解为一个很长的一维数组。

   测试图像(rain-images-idx3-ubyte)与训练图像(train-images-idx3-ubyte)由5部分组成：

| 32bits int(magic number) | 32bits int图像个数 | 32bits int图像高度28 | 32bits int图像宽度28 | 像素值（pixels） |
| ------------------------ | -------------- | ---------------- | ---------------- | ----------- |
|                          |                |                  |                  |             |

​	测试标签(t10k-labels-idx1-ubyte)与训练标签(train-labels-idx1-ubyte)由3部分组成：

| 32bits int(magic number) | 32bits int图像个数 | 标签（labels） |
| ------------------------ | -------------- | ---------- |
|                          |                |            |

5. (图中红线为tanh,绿线为cosh,蓝线为sinh)

![tanh](http://blog.mathteachersresource.com/wp-content/uploads/2015/10/hyperbolicheadfig.jpg)

