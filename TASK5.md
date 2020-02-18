#  Task5

##  1 卷积神经网络基础

###  1.1 互相关运算和卷积

二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏置来得到输出。卷积层的模型参数包括卷积核和标量偏置。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\8.PNG)

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\9.PNG)

###  1.2 特征图与感受野

二维卷积层输出的二维数组可以看作是输入在空间维度（宽和高）上某一级的表征，也叫特征图（feature map）。影响元素$x$的前向计算的所有可能输入区域（可能大于输入的实际尺寸）叫做$x$的感受野（receptive field）。

以图1为例，输入中阴影部分的四个元素是输出中阴影部分元素的感受野。我们将图中形状为$2 \times 2$的输出记为$Y$，将$Y$与另一个形状为$2 \times 2$的核数组做互相关运算，输出单个元素$z$。那么，$z$在$Y$上的感受野包括$Y$的全部四个元素，在输入上的感受野包括其中全部9个元素。可见，我们可以通过更深的卷积神经网络使特征图中单个元素的感受野变得更加广阔，从而捕捉输入上更大尺寸的特征。

## 2 LeNet

结构如图所示：

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\10.PNG)



```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(
    batch_size=batch_size, root='/home/kesci/input/FashionMNIST2065')
print(len(train_iter))
```

输出为：

```python
235
```

即一共有235组数据，每组数据中有256张图片

==代码部分已经放到了github中，里面添加了我在学习过程中的理解的注释，用到的d2lzh_pytorch包以及fashion_mnist数据集也已经打包上传到了GitHub中。==

##  3 卷积神经网络深入

###  3.1 深度卷积神经网络（AlexNet）

LeNet:  在大的真实数据集上的表现并不尽如⼈意。
 1.神经网络计算复杂。
 2.还没有⼤量深⼊研究参数初始化和⾮凸优化算法等诸多领域。

机器学习的特征提取:手工定义的特征提取函数
 神经网络的特征提取：通过学习得到数据的多级表征，并逐级表⽰越来越抽象的概念或模式。

神经网络发展的限制:数据、硬件

####  3.1.1 网络结构

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\14.PNG)

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\12.PNG)

从下到上：输入层、卷积层、全连接隐藏层、全连接输出层。

括号中的数代表通道数，可以看到右边比左边基本上多了数十倍，即特征数多。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\13.PNG)

###  3.2 重复使用元素的网络（VGG）

虽然AlexNet指明了深度卷积神经网络可以取得出色的结果，但并没有提供简单的规则以指导后来的研究者如何设计新的网络。而VGG提出了可以通过重复使用简单的基础块来构建深度模型的思路。

VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为$3\times 3$的卷积层后接上一个步幅为2、窗口形状为$2 \times2$的最大池化层。卷积层保持输入的高和宽不变，而池化层则对其减半。

结构如图所示：

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\15.PNG)

### 3.3 网络中的网络（NiN）

LeNet、AlexNet和VGG在设计上的共同之处是：==先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果。==

AlexNet和VGG对LeNet的改进主要在于如何对这两个模块加宽（增加通道数）和加深。

NiN：串联多个由卷积层和“全连接”层构成的小⽹络来构建⼀个深层网络。⽤了输出通道数等于标签类别数的NiN块，然后使⽤全局平均池化层对每个通道中所有元素求平均并直接⽤于分类。

卷积层的输入和输出通常是四维数组（样本，通道，高，宽），而全连接层的输入和输出则通常是二维数组（样本，特征）。如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为四维。

$1 \times 1$卷积层可以看成全连接层，其中空间维度（高和宽）上的每个元素相当于样本，通道相当于特征。因此，NiN使用$1 \times 1$卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\16.PNG)

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\17.PNG)

NiN块是NiN中的基础块。它由一个卷积层加两个充当全连接层的$1 \times 1$卷积层串联而成。其中第一个卷积层的超参数可以自行设置，而第二和第三个卷积层的超参数一般是固定的。

==1×1卷积核作用:==

1. 放缩通道数：通过控制卷积核的数量达到通道数的放缩。
2. 增加非线性。1×1卷积核的卷积过程相当于全连接层的计算过程，并且还加入了非线性激活函数，从而可以增加网络的非线性。
3. 计算参数少。

NiN重复使⽤由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层网络。NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。

###  3.4 GoogLeNet

GoogLeNet中的基础卷积块叫作Inception块。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\18.PNG)

GoogLeNet的完整结构：

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\19.PNG)

输入图像是$1\times96\times96$

