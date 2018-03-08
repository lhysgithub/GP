[TOC]

##### 2017/12/26 

一切终结结束，一切又会有新的开始。从今天起，我将纵身于学术，一去不复返。

学习TensorFlow官方文档中文版基本使用方法。

最终还是玩儿过去了~

##### 2017/12/27

代码中 assign() 操作是图所描绘的表达式的一部分, 正如 add() 操作一样. 所以在调用 run() 执行表达式之前, 它并不会真正执行赋值操作.

我选择去看TensorFlow英文文档。

通读[GetStarted](https://www.tensorflow.org/get_started/get_started)文档

如果需要更多的编码需求参考[ProgrammerGuide](https://www.tensorflow.org/programmers_guide)文档

本次学习重要心得就是中文文档看不懂一定要去读英文文档，中文的文档无论是逻辑排列还是词汇使用都不适合理解，这或许和我们高中做了三年的阅读理解有关，这样还能顺便学习一下不认识的单词。

##### 2018/01/15

确立了使用遗传算法作为搜索算法

确立了使用通过找到特征代表的方式可解释的选取修改的像素点

确立了在搜索时引入置信因子的思想

##### 2018/01/20

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



##### 2018/01/22

计划：

1. 完成基础遗传算法的实现
2. 完成一片托福试卷的练习

实行：

1. [Markdown文档](http://wowubuntu.com/markdown/#p)
2. [numpy.argmax](http://www.cnblogs.com/zhouyang209117/p/6512302.html)
3. [MINIST数据集格式](https://www.jianshu.com/p/84f72791806f)
4. [tf.reduce_sum](https://www.zhihu.com/question/51325408)
5. [使用转换mnist数据库保存为bmp图片](http://blog.csdn.net/u010194274/article/details/50817999)
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
6. [python中`_`、`__`和`__xx__`的区别](http://www.cnblogs.com/coder2012/p/4423356.html)

  `_`表示私有，不是API不应该从外界调用(在外界能用_Method/Attribute调用)

  `__`表示不能被子类重载（在外界中能用_Class__Method/Attribute调用）

  `__xx__`表示该方法是被系统使用（可用来重载运算符，外界调用方法为Method（Object）而不是传统的Object.Method()）

  这些都是为了实现面向对象编程的某些特性。

  待看：

- [1710.06081] Boosting Adversarial Attacks with Momentum（清华攻击）

  | **TOEFL** | （点击链接即可下载）                               |
  | --------- | ---------------------------------------- |
  | **机经：**   | **https://pan.baidu.com/s/1pKF8IEn**     |
  | **考位：**   | **https://pan.baidu.com/s/1jIN0Fg6**     |
  | **模考软件：** | **<https://pan.baidu.com/s/1nuTk9y5>[https://pan.baidu.com/s/1c4m4qkw托福高频词汇](https://pan.baidu.com/s/1c4m4qkw%E6%89%98%E7%A6%8F%E9%AB%98%E9%A2%91%E8%AF%8D%E6%B1%87)** |
  | **最新资料：** | **https://pan.baidu.com/s/1dE9N3HZ**     |

##### 2018/01/23

对于2018/01/22总结：浪费了很多时间在无关紧要的事情上，最主要的任务并没有完成。而且晚上的时候还浪费了些时间在荒野行动上。这样是不被允许的，对于无底洞式的东西，尽量远离。



对于2018/01/23日上午的批评：浪费了很多时间，包括看小说、看漫画等，千万不要返回看小说的坑！本来应该用作学习托福的时间却被用来玩耍，朕甚是心痛！下不为例！



对此做出解决方案：每天除了FGO拒绝玩儿其他的游戏。每天都是尽可能的去学习！而不是尽可能的去玩耍！



对于2018/01/23的计划：

1. 攻击遗传算法的实现

2. 针对Cifer数据集图片进行图像分割

3. 1套托福题

   次要计划：

   1. 寻找能够增加简历的内容
   2. 投简历
   3. 剪头发



2018/01/22 的时候你就没能实现计划，今天，拼尽全力去实现吧。

千万不能磨洋工！

针对遗传攻击算法：我们为了顺应时代的潮流，我们打算使用cleverhans库作为基础支撑库，即，我们将为cleverhans增加一种新的攻击方法，即遗传攻击！

由于一开始想要使用CW攻击的图像分类器生成、对抗样本标签生成等非攻击方法部分的代码，所以虽然对于这方面算法有了比较清晰的了解，但是，现在又要去大概了解一下cleverhans库中这些非攻击方法部分的代码。任务艰巨！

参照Attacks.class CarliniWagnerL2(Attack):进行GAAttack的编写。

提出疑问：

1. 为什么要逆双曲正弦？
2. Tensorflow的计算图建立起来后是怎么运行的？！

学习过程：

1. [numpy.ones](http://python.usyiyi.cn/documents/NumPy_v111/reference/generated/numpy.ones.html)

2. [Python enumerate() 函数](http://www.runoob.com/python/python-func-enumerate.html)

3. [numpy.arctanh](http://python.usyiyi.cn/documents/NumPy_v111/reference/generated/numpy.arctanh.html)

4. [Python_zip() 函数](http://www.runoob.com/python/python-func-zip.html)

5. CWAttack中超参C为通关过二分法对于每一个图片自行学习的。

6. 18：31看完CW攻击源码！完全！

7. 明天进行GA算法的实现！


##### 2018/01/24

今日计划：

1. GA算法完成30%
2. 托福3小时



今日感受：

1. 一心二用是不好的（在你精神力没有那么强大的时候），只会使得两件事情都干不好！（所以别再边玩儿边学了！）
2. 中午一定要睡觉，不然学习玩耍什么的都不会很舒服。



学习心得：

1. tensorflow并非并行计算！
2. [numpy.clip()](http://blog.csdn.net/Light_blue_love/article/details/41897799)
3. [python.set()](http://blog.csdn.net/business122/article/details/7541486)
4. ​

GA改造工程：

1. def show(img) 可以将本函数改造成将数据存储在图片文件中！！！
2. 在计算图中加入GA算法，我需要加入一个新的optimizer

今日进度：

1. 不再使用cleverhans库作为基础代码

##### 2018/01/26

今日组会，定下任务：

1. 2月3日前完成实验：
   1. 遗传算法
   2. 最小图像特征导引
2. 2月28日前完成中文研究性论文：
   1. 综述（开题和论文北京用）
   2. 实验简述
   3. latex


##### 2018/01/30

已经一月三十日了，不能在玩耍了，好好学习，天天向上。

今日任务：

1. 把图片整理好
2. 把遗传算法写完
3. 托福两篇阅读，1小时单词
4. 


##### 2018/02/01

我，逍遥，愿随大圣一战

我，明眸，愿随大圣一战

如今，兄弟们走着走着就散了，我本以为以后还会遇到那帮兄弟，但是，最后只有我落寞一个人。

或许是因为习惯了一个人，一开始就将自己保护起来，谁人都不知，谁人都不晓。

游戏中也不能释放自我的话，那游戏又有何用。。。

我决定，我要融入魔域这个圈子，没有办法，魔域终究是执念。

路还长，慢慢走。

今后，我将不再浪费我的时光，讨好他人，对不起，做不到。

今天制定一个完美的时间表。

如今，我关注四个游戏。

人生！魔域！FGO！农药！

农药利用模拟器和手机联动快速完成每日任务，大概一小时。

FGO上线10分钟下线10小时。

魔域一条龙1小时，然后开启上线3分钟下线半小时的进程。

人生，剩下的时间全是留给人生的。

所以，每日游戏时间为大概4小时，睡眠时间为6小时，吃饭时间为1小时，剩下时间应该都在学习！

##### 2018/02/03

任务制定：

白天：学习英语

1. 单词/词组 300词
2. 阅读3篇

晚上：毕业设计

1. 完成GA算法

间歇可以玩玩儿魔域。

##### 2018/02/08

前情总结：贪玩，不负责任。

到了2月9号了，你已经玩儿了近一个月了，该付些责任了~

今后的每一天，你都要加倍努力才行，不然，GG。

你的计划：毕设论文、托福、学车。

##### 2018/02/11

今天不做太多计划，只做记录。

##### 2018/02/14

英语待解决问题：

1. 重读发音问题。authoritative 和 author

##### 2018/02/25

昨日返校，今日开始疯狂肝毕设。

今晚必须完成GA攻击的代码，拼死！

##### 2018/02/28

27号和28号帮助老师写基金申请书。

今天定一下8学期第一周（02/26-03/04）的任务：

毕业设计：彻底完成遗传算法的实现，能跑，能实验。

托福：每日单词100+每日阅读1篇+每日听力1篇+每日口语1篇。

##### 2018/03/01

今日背单词30，复习单词60。。。

GA重构差优化器部分。

##### 2018/03/03

英语待解决问题：

1.  collaborative 与 labor a的发音不同问题。

目前面临的严峻问题：
1. 单词背的太慢，严重拖延进度，严重侵占其他时间。
2. 阅读训练没时间、毕设代码没时间、论文阅读没时间。（前两项比较重要）

解决策略：
1. 进行高效的背单词方式，并规定背单词的时间。

背单词的时间限定：睡醒后1小时，不能再多了！以及晚上睡前半小时。
口语练习：睡前另半小时留给口语自练。
听力与阅读练习给出大量时间练习。
（背单词采用边走边背的方式，并且可以在吃法时候与闲暇的时候背。）

毕设代码也给出大量时间练习。

##### 2018/03/05

今天上午花1.5小时完成100单词的复习

今天中午花2小时完成张老师的任务

今天下午花1小时更谭BOSS谈话，改变之前的毕设题目

今天晚上花5小时得到Andrew Ilyas的代码不能跑，需要修改，修改幅度未知。

明天准备跑一下学姐发的代码

现在是22：40我打算再学学英语12点左右回宿舍。

##### 2018/03/06

今天的活动比较规律：

睡醒后1小时背单词成功在睡了三次之后背了100词

成功在本地重现Query-efficent black-box Adversarial Example（下午2点至5点半和7点至9点）

接下来，学英语阅读或听力！

##### 2018/03/08

今日任务：

1. 接口化腾讯AI的API

2. 明日ppt（讲解内容，指定目标）

3. 托福听力、单词

4. 与基友晚宴

5. 开题报告，完整写完


已解决的问题：

1. 基于query-efficiency black attack攻击实现本地黑盒攻击inceptionV3
2. 完成腾讯物体识别API的python实现

待解决的问题：

1. 服务器之前一直用CPU跑代码，现在要安装GPU驱动，但是现在服务器安装不上东西
2. 服务器不能连外网（我使用PuTTY+Xing server 利用ssh转发X11来解决，但是。。。目前还不行）
3. ​

 