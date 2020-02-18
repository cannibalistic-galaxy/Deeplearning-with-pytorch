#  Task3

##  1 模型选择、过拟合、欠拟合

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\2.PNG)

###  1.1 代码解析

（1）

```python
torch.randn(*sizes, out=None) → Tensor
```

返回一个张量，包含了从标准正态分布(均值为0，方差为 1，即高斯白噪声)中抽取一组随机数，形状由可变参数`sizes`定义。

例如：

```python
>>> torch.randn(2, 3)

 1.4339  0.3351 -1.0999
 1.5458 -0.9643 -0.3558
[torch.FloatTensor of size 2x3]
```

（2）如果我们有两个tensor是A和B，想把他们拼接在一起，需要如下操作：

```python
C = torch.cat( (A,B),0 )  #按维数0拼接（竖着拼）

C = torch.cat( (A,B),1 )  #按维数1拼接（横着拼）
```

（3）

```python
torch.pow(input, exponent, out=None)
```

对输入input按元素求exponent次幂值，并返回结果张量，幂值exponent可以为标量也可以是和input相同大小的张量。

（4）

```python
 torch.nn.Linear（in_features，out_features，bias = True ）
```

是对输入的数据的一种线性变换，

- **in_features** – 每个输入样本的尺寸
- **out_features** – 每个输出样本的尺寸

为代码添加注释后：

```python
n_train, n_test, true_w, true_b  = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1)) # 生成(200，1)的服从正态分布的x
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1) # 将x，x^2, x^3按横向拼接
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b) # 即y
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 加上噪声
```

```python
def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1) 
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    batch_size = min(10, train_labels.shape[0])    # 批量大小=10和行长度的最小值
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 梯度下降法
    train_ls, test_ls = [], []
    for _ in range(num_epochs):  # 训练次数
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1)) # 计算损失函数
            optimizer.zero_grad()  # 梯度清零
            l.backward()  #反向传播计算梯度
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)
```

###  1.2 权重衰减

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\3.PNG)

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\4.PNG)

迭代过程中后面的项的系数不变，前面项的系数衰减了。

###  1.3 丢弃法（dropout）

本文介绍的是倒置丢弃法（inverted dropout）

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\5.PNG)

注意，这里的超参数$\xi$的取值为0或1。

断言：

```python
assert expression
```

如果expression是True，则继续执行下面内容，若为False，则直接报错，等价于

```python
if not expression:
    raise AssertionError
```

当有两层隐藏层的时候，丢弃率设置两个，代码例子如下：

```python
drop_prob1, drop_prob2 = 0.2, 0.5 # 两层隐藏层

def net(X, is_training=True): 
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, W1) + b1).relu()
    if is_training:  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, W2) + b2).relu()
    if is_training:
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return torch.matmul(H2, W3) + b3
```

##  2 梯度消失、梯度爆炸

###  2.1 定义

深度模型有关数值稳定性的典型问题是消失（vanishing）和爆炸（explosion）。

**当神经网络的层数较多时，模型的数值稳定性容易变差。**

假设一个层数为$L$的多层感知机的第$l$层$\boldsymbol{H}^{(l)}$的权重参数为$\boldsymbol{W}^{(l)}$，输出层$\boldsymbol{H}^{(L)}$的权重参数为$\boldsymbol{W}^{(L)}$。为了便于讨论，不考虑偏差参数，且设所有隐藏层的激活函数为恒等映射（identity mapping）$\phi(x) = x$。给定输入$\boldsymbol{X}$，多层感知机的第$l$层的输出$\boldsymbol{H}^{(l)} = \boldsymbol{X} \boldsymbol{W}^{(1)} \boldsymbol{W}^{(2)} \ldots \boldsymbol{W}^{(l)}$。此时，如果层数$l$较大，$\boldsymbol{H}^{(l)}$的计算可能会出现衰减或爆炸。举个例子，假设输入和所有层的权重参数都是标量，如权重参数为0.2和5，多层感知机的第30层输出为输入$\boldsymbol{X}$分别与$0.2^{30} \approx 1 \times 10^{-21}$（消失）和$5^{30} \approx 9 \times 10^{20}$（爆炸）的乘积。当层数较多时，梯度的计算也容易出现消失或爆炸。

###  2.2 随机初始化模型参数

假设输出层只保留一个输出单元$o_1$（删去$o_2$和$o_3$以及指向它们的箭头），且隐藏层使用相同的激活函数。如果将每个隐藏单元的参数都初始化为相等的值，那么==在正向传播时每个隐藏单元将根据相同的输入计算出相同的值，并传递至输出层==。在反向传播中，每个隐藏单元的参数梯度值相等。因此，这些参数在使用基于梯度的优化算法==迭代后值依然相等==。之后的迭代也是如此。在这种情况下，无论隐藏单元有多少，隐藏层本质上==只有1个隐藏单元==在发挥作用。因此，正如在前面的实验中所做的那样，我们通常将神经网络的模型参数，特别是权重参数，进行随机初始化。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\6.PNG)

随机初始化模型参数的方法有很多。在线性回归的简洁实现中，我们使用`torch.nn.init.normal_()`使模型`net`的权重参数采用正态分布的随机初始化方式。不过，PyTorch中`nn.Module`的模块参数都采取了较为合理的初始化策略（不同类型的layer具体采样的哪一种初始化方法的可参考[源代码](https://github.com/pytorch/pytorch/tree/master/torch/nn/modules)），因此一般不用我们考虑。

还有一种比较常用的随机初始化方法叫作Xavier随机初始化。
假设某全连接层的输入个数为$a$，输出个数为$b$，Xavier随机初始化将使该层中权重参数的每个元素都随机采样于均匀分布

$$
U\left(-\sqrt{\frac{6}{a+b}}, \sqrt{\frac{6}{a+b}}\right).
$$


它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。

###  2.3 协变量偏移

这里我们假设，虽然输入的分布可能随时间而改变，但是标记函数，即条件分布P（y∣x）不会改变。在一个看起来与测试集有着本质不同的数据集上进行训练，而不考虑如何适应新的情况，这是不是一个好主意。不幸的是，这是一个非常常见的陷阱。

统计学家称这种协变量变化是因为问题的根源在于特征分布的变化（即协变量的变化）。数学上，我们可以说P（x）改变了，但P（y∣x）保持不变。尽管它的有用性并不局限于此，当我们认为x导致y时，协变量移位通常是正确的假设。

###  2.4 标签偏移

当我们认为导致偏移的是标签P（y）上的边缘分布的变化，但类条件分布是不变的P（x∣y）时，就会出现相反的问题。当我们认为y导致x时，标签偏移是一个合理的假设。例如，通常我们希望根据其表现来预测诊断结果。在这种情况下，我们认为诊断引起的表现，即疾病引起的症状。有时标签偏移和协变量移位假设可以同时成立。例如，当真正的标签函数是确定的和不变的，那么协变量偏移将始终保持，包括如果标签偏移也保持。有趣的是，当我们期望标签偏移和协变量偏移保持时，使用来自标签偏移假设的方法通常是有利的。这是因为这些方法倾向于操作看起来像标签的对象，这（在深度学习中）与处理看起来像输入的对象（在深度学习中）相比相对容易一些。

病因（要预测的诊断结果）导致 症状（观察到的结果）。  

训练数据集，数据很少只包含流感p(y)的样本。  

而测试数据集有流感p(y)和流感q(y)，其中不变的是流感症状p(x|y)。

###  2.5 概念偏移

在概念转换中，即标签本身的定义发生变化的情况。

三者个人理解为：==协变量偏移是x改变，标签偏移是y改变 概念偏移是y的定义改变==

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\7.PNG)

###  2.6 Kaggle比赛

流程：读取数据集->输出部分数据集以观察格式->筛选能够用于训练的特征->数据预处理：对连续数值的特征标准化、将离散数值转成指示特征、转化为Numpy格式的数据，并转成Tensor->模型设计

##  3 循环神经网络进阶

RNN的结构：

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\21.PNG)

问题：容易出现梯度衰减或梯度爆炸。

###  3.1 门控循环神经网络（GRU）：

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\22.PNG)

-  重置⻔有助于捕捉时间序列⾥短期的依赖关系；
-  更新⻔有助于捕捉时间序列⾥⻓期的依赖关系。

需要初始化的参数：一共9+2+1=12个参数。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\23.png)

###  3.2 长短期记忆（LSTM）long short-term memory

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\24.PNG)

输出门是输出到隐藏状态的流动，其他的门都是到输出的状态流动。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\25.PNG)

参数：==注意此处的$C_{-1}$也是要初始化的，图上忘记标注了==。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\26.PNG)

###  3.3 深度循环神经网络

结构如图：

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\27.PNG)

```python
gru_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens,num_layers=2)
```

只要令num_layers=2，就能得到一个二层的循环神经网络。即得到如下图所示红框部分组成的网络：

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\28.PNG)

###  3.4 双向循环神经网络

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\29.PNG)

```python
gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens,bidirectional=True)
```

令bidirectional=True即可实现双向循环神经网络。

![](E:\Github\GithubProject\DeeplearningWithPytorch\笔记\TASK3&TASK4&TASK5\图片\30.PNG)