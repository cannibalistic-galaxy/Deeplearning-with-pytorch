#  Task1 & Task2

##  1 线性回归

###  1.1 框架

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/1.PNG)

###  1.2 线性回归的过程

####  1.2.1 模型定义

以房屋价格预测为例。

==应用目标：==预测一栋房子的售出价格（y）；

==影响因素：==面积（$x_1$）、房龄($x_2$)；

==模型：==$\hat{y}=x_1w_1+x_2w_2+b$

$w_1$ 和 $w_2$ 是权重（weight），$b$ 是偏差（bias），均为标量。它们是线性回归模型的参数（parameter）。模型输出$\hat{y}$是线性回归对真实价格$y$的预测或估计。我们通常允许它们之间有一定误差。

####  1.2.2 模型训练

#####  （1）模型训练

通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小的过程。

==训练数据集（training data set）/训练集（training set）：==多栋房屋的真实售出价格和它们对应的面积和房龄；

==一个样本（sample）：==一栋房屋；

==标签（label）：==真实售出价格；

==特征（feature）：==用来预测标签的两个因素，表征样本的特点。

设采集的样本数为$n$，索引为$i$的样本特征为$x_1^{(i)}$和 $x_2^{(i)}$，标签为$y^{(i)}$。对于索引为$i$的房屋，预测表达式为：
$$
\hat{y}^{(i)}=x_1^{(i)}w_1+x_2^{(i)}w_2+b
$$

#####  （2）损失函数

在模型训练中，我们需要衡量价格预测值与真实值之间的误差。通常我们会选取一个非负数作为误差，且数值越小表示误差越小。平方误差函数在评估索引为$i$时的样本误差表达式为：
$$
l^{(i)}(w_1,w_2,b)=\frac{1}{2}(\hat{y}^{(i)}-y^{(i)})^2
$$
其中常数$\frac{1}{2}$使对平方项求导后的常数系数为1（没有实质作用，使形式简单一点）。

通常，我们用训练数据集中所有样本误差的平均来衡量模型预测的质量，即
$$
l(w_1,w_2,b)=\frac{1}{n}\sum_{i=1}^nl^{(i)}(w_1,w_2,b)
$$

#####  （3）优化算法

当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）。大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）。

小批量随机梯度下降（mini-batch stochastic gradient descent）在深度学习中被广泛使用。

==算法流程：==

==先选取一组模型参数的初始值，如随机选取；接下来对参数进行多次迭代，使每次迭代都可能降低损失函数的值。在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）$B$，然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。==

==梯度使指损失函数关于权重参数的导数。==

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/5.PNG)

在上式中，$∣B∣$ 代表每个小批量中的样本个数（批量大小，batch size），$η$ 称作学习率（learning  rate）并取正数。需要强调的是，这里的批量大小和学习率的值是人为设定的，并不是通过模型训练学出的，因此叫作超参数（hyperparameter）。我们通常所说的“调参”指的正是调节超参数，例如通过反复试错来找到超参数合适的值。

####  1.2.3 模型预测

模型训练完成后，我们将模型参数 $w_1,w_2,b$ 在优化算法停止时的值分别记作 $\hat{w}_1,\hat{w}_2,\hat{b}$。注意，这里我们得到的并不一定是最小化损失函数的最优解$ w^*_1,w^*_2,b^*$，而是对最优解的一个近似。然后，我们就可以使用学出的线性回归模型 $x_1\hat{w}_1+x_2\hat{w}_2+\hat{b}$来估算训练数据集以外任意一栋面积（平方米）为$x_1$、房龄（年）为$x_2$的房屋的价格了。这里的估算也叫作模型预测、模型推断或模型测试。

因为本例中的代码和第二节Softmax本质差别不大，因此代码笔记放在第二节记录。

###  1.3 打卡作业题：

题目1：

==题目==：假如你正在实现一个全连接层，全连接层的输入形状是7×87 $\times$ 7×8，输出形状是7×1，其中7是批量大小，则权重参数w和偏置参数b的形状分别是____和____。

==解答==：注意7是批量大小，此全连接层应如图所示

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/2.PNG)

题目2：

在pytorch里，view函数相当于Numpy中的reshape

```python
x.view(-1,4) #不确定要几行，但是要4列
```

##  2 Softmax

### 2.1 基本概念

整体思路如下：

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/4.PNG)

softmax层是一个单层神经网络：

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/3.PNG)

softmax的核心公式：
$$
\hat{y}_1, \hat{y}_2, \hat{y}_3=softmax(o_1,o_2,o_3)
$$

$$
\hat{y}_1=\frac{exp(o_1)}{\sum_{i=1}^{3}exp(o_i)}, \hat{y}_2=\frac{exp(o_2)}{\sum_{i=1}^{3}exp(o_i)}, \hat{y}_3=\frac{exp(o_3)}{\sum_{i=1}^{3}exp(o_i)}
$$

可以看出$\hat{y}_1+\hat{y}_2+\hat{y}_3=1$且$0\leq \hat{y}_1, \hat{y}_2, \hat{y}_3 \leq 1$，即$\hat{y}_1, \hat{y}_2, \hat{y}_3$为合法概率分布。softmax运算不改变预测类别输出。

###  2.2 代码实现

#### 2.2.1 用到的一些函数

```
class torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)
```

- root（string）– 数据集的根目录，其中存放processed/training.pt和processed/test.pt文件。
- train（bool, 可选）– 如果设置为True，从training.pt创建数据集，否则从test.pt创建。
- download（bool, 可选）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
- transform（可被调用 , 可选）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：transforms.RandomCrop。
- target_transform（可被调用 , 可选）– 一种函数或变换，输入目标，进行变换。

```python
torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, num_works=0, drop_last=False)
```

功能：构建可迭代的数据装载器  

- dataset: Dataset类，决定数据从哪读取及如何读取
- batchsize : 批大小
- num_works: 是否多进程读取数据，例如设置num_works=4，则为4个进程读取数据
- shuffle: 每个epoch是否乱序
- drop_last：当样本数不能被batchsize整除时，是否舍弃最后一批数据  

==Epoch、Iteration、Batchsize的关系：==

- Epoch: 所有训练样本都已输入到模型中，称为一个Epoch
- Iteration：一批样本输入到模型中，称之为一个Iteration
- Batchsize：批大小，决定一个Epoch有多少个Iteration。此例中的多少批就可以理解成多少张图片  

对多维Tensor按维度操作：

```python
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征
```

输出：

```
tensor([[5, 7, 9]])
tensor([[ 6],
        [15]])
tensor([5, 7, 9])
tensor([ 6, 15])
```

gather是在一个tensor中取数据，其使用方式如下：

```
torch.gather(input, dim, index, out=None)
```

其中，dim是维度，index是索引，dim=1代表按行取，此时索引就是列号，如下实例：

```
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
```

其中y.view(-1,1)得到的是[2, 1]的tensor，即，取第一行第0个和第二行第2个。因此==输出==为

```
tensor([[0.1000],
        [0.5000]])
```

####  2.2.2 代码详解

训练的代码如下：

```python
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
```

首先对代码中输入的参数进行分析:

##### （1）net为softmax回归模型，即：

```
softmax(线性回归模型)
```

```python
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```

#####  （2） train_iter、test_iter为训练数据集和测试数据集

```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

#####  （3）loss为损失函数，此例中为交叉熵损失函数

```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
```

#####  （4）num_epochs, batch_size, params, lr均为定义的数据：

```
num_epochs, lr = 5, 0.1
batch_size = 256
params = [W, b]
```

#####  （5）optimizer=None，则采用的是小批量随机梯度下降

```python
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
```

然后分析后面的代码：

训练5次，X代表的是特征，y是标签。y_hat代表预测标签，l计算损失函数。通过反向传播（backward）求解梯度，grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。然后通过梯度下降法进行参数更新。最后计算训练和测试的准确率。



##  4 文本预处理

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/6.PNG)

###  4.1 读入文本

用到的函数：

```
re.sub(pattern, repl, string, count=0, flags=0)
```

- pattern：表示正则表达式中的模式字符串；
- repl：被替换的字符串（既可以是字符串，也可以是函数）；
- string：要被处理的，要被替换的字符串；
- count：匹配的次数, 默认是全部替换
- flags：具体用处不详

```python
line.strip() #去掉前缀的、后缀的空白字符，包括空格和换行符（只能删除开头或者结尾的字符，不能删除中间部分的字符）
```

代码分析：

```
import collections
import re

def read_time_machine():
    with open('/home/kesci/input/timemachine7163/timemachine.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines


lines = read_time_machine()
print('# sentences %d' % len(lines))
```

其中每次读取f中的一行文件，然后把这行文件中的前缀、后缀的空格和换行符删除，并全部转化为小写字母。再将每行中不是[a-z]字母的字符串替换成空格，即把非英文字符的字符串全部替换成空格，得到lines列表。

###  4.2 分词

对每个句子进行分词，也就是将一个句子划分成若干个词（token），转换为一个词的序列。

代码分析：

```python
def tokenize(sentences, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences] # 用空格做分隔符进行分隔
    elif token == 'char':
        return [list(sentence) for sentence in sentences]  # 将句子直接转换成列表
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)
tokens[0:2]
```

传入参数：

sentences表示传入的句子，token表示做哪个级别的分词，是'word'还是'char'。

###  4.3 建立字典

为了方便模型处理，我们需要将字符串转换为数字。因此我们需要先构建一个字典（vocabulary），将每个词映射到一个唯一的索引编号。

代码分析：

```python
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # 统计词频，counter是个字典，词：词频
        self.token_freqs = list(counter.items())  #变成列表，即对语料库中进行去重
        self.idx_to_token = []  
        if use_special_tokens:  # 添加一些需要的内容
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token] #把语料库中的且没有在idx_to_token出现的词添加到idx_to_token中
        self.token_to_idx = dict()  #从词到索引号的字典
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx  # 把词和词的下标添加到idx_to_token中

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):  #统计词频的函数
    tokens = [tk for st in sentences for tk in st]  #展开，得到一维列表
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数
```

参数分析：tokens就是上面tokenize函数返回的值；min_freq为一个阈值，当某个词出现次数小于这个阈值时，就把它忽略掉；use_special_tokens是一个标志，是否要使用特殊的token。

###  4.4 将词转化为索引

使用字典，我们可以将原文本中的句子从单词序列转换为索引序列。

```
for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

### 4.5 现有工具分词

前面介绍的分词方式非常简单，它至少有以下几个缺点:

1. 标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
2. 类似“shouldn't", "doesn't"这样的词会被错误地处理
3. 类似"Mr.", "Dr."这样的词会被错误地处理

我们可以通过引入更复杂的规则来解决这些问题，但是事实上，有一些现有的工具可以很好地进行分词，我们在这里简单介绍其中的两个：[spaCy](https://spacy.io/)和[NLTK](https://www.nltk.org/)。

```
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
print([token.text for token in doc])

from nltk.tokenize import word_tokenize
from nltk import data
data.path.append('/home/kesci/input/nltk_data3784/nltk_data')
print(word_tokenize(text))
```

##  5 语言模型

语言模型（language model）是自然语言处理的重要技术。自然语言处理中最常见的数据是文本数据。可以把一段自然语言文本看作一段离散的时间序列。

假设一段长度为$T$的文本中词依次为$w_1, w_2, ..., w_T$，那么在离散的时间序列中，$w_t(1\leq t \leq T)$可以看作在时间步$t$的输出活标签。给定一段长度为$T$的词的序列$w_1, w_2, ..., w_T$，语言模型将计算该序列的概率：
$$
P(w_1, w_2, ..., w_T)
$$

###  5.1 语言模型的计算

假设序列$w_1, w_2, ..., w_T$中的每个词是依次生成的，则有：
$$
P(w_1, w_2, \ldots, w_T)
= \prod_{t=1}^T P(w_t \mid w_1, \ldots, w_{t-1})\\
= P(w_1)P(w_2 \mid w_1) \cdots P(w_T \mid w_1w_2\cdots w_{T-1})
$$
例如，一段含有4个词的文本序列的概率：
$$
P(w_1, w_2, w_3, w_4) =  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_1, w_2, w_3).
$$

为了计算语言模型，我们需要计算词的概率，以及一个词在给定前几个词的情况下的条件概率，即语言模型参数。设训练数据集为一个大型文本语料库，如维基百科的所有条目。词的概率可以通过该词在训练数据集中的相对词频来计算。例如，$w_1$的概率可以计算为：
$$
\hat{P}(w_1)=\frac{n(w_1)}{n}
$$
其中，$n(w_1)$表示语料库中以$w_1$作为第一个词的文本数量，$n$为语料库中文本的总数量。

类似的，给定$w_1$情况下，$w_2$的条件概率可以计算为：


$$
\hat P(w_2 \mid w_1) = \frac{n(w_1, w_2)}{n(w_1)}
$$

其中$n(w_1, w_2)$为语料库中以$w_1$作为第一个词，$w_2$作为第二个词的文本的数量。

###  5.2 n元语法

$n$元语法通过马尔可夫假设（虽然并不一定成立）简化了语言模型的计算。

马尔科夫假设是指一个词的出现只与前面$n$个词相关，即==$n$==阶马尔可夫链（Markov chain of order $n$），如果$n=1$，那么有$P(w_3 \mid w_1, w_2) = P(w_3 \mid w_2)$。

基于==$n-1$==阶马尔可夫链（注意不是$n$阶马尔可夫），我们可以将语言模型改写为


$$
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T P(w_t \mid w_{t-(n-1)}, \ldots, w_{t-1}) .
$$
以上也叫==$n$元语法==（$n$-grams），它是基于==$n - 1$==阶马尔可夫链的概率语言模型。例如，当$n=2$时，含有4个词的文本序列的概率就可以改写为：（注意$n$元语法是基于$n-1$阶马尔科夫链的）
$$
\begin{align*}
P(w_1, w_2, w_3, w_4)
&= P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_1, w_2, w_3)\\
&= P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_2) P(w_4 \mid w_3)
\end{align*}
$$
当$n$分别为1、2和3时，我们将其分别称作一元语法（unigram）、二元语法（bigram）和三元语法（trigram）。例如，长度为4的序列$w_1, w_2, w_3, w_4$在一元语法、二元语法和三元语法中的概率分别为


$$
\begin{aligned}
P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2) P(w_3) P(w_4) ,\\
P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_2) P(w_4 \mid w_3) ,\\
P(w_1, w_2, w_3, w_4) &=  P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_1, w_2) P(w_4 \mid w_2, w_3) .
\end{aligned}
$$

###  5.3 代码分析

####  5.3.1 读取数据集并建立字符索引

代码注释：

```python
def load_data_jay_lyrics():
    with open('/home/kesci/input/jaychou_lyrics4703/jaychou_lyrics.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ') #替换换行符为空格
    corpus_chars = corpus_chars[0:10000]  #保留前10000个字符
    idx_to_char = list(set(corpus_chars)) # 去重，得到索引到字符的映射
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)]) # 字符到索引的映射
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars] # 将每个字符转化为索引，得到一个索引的序列
    return corpus_indices, char_to_idx, idx_to_char, vocab_size
```

####  5.3.2 时序数据的采样

在训练中我们需要每次随机读取小批量样本和标签。与之前章节的实验数据不同的是，时序数据的一个样本通常包含连续的字符。假设时间步数为5，样本序列为5个字符，即“想”“要”“有”“直”“升”。该样本的标签序列为这些字符分别在训练集中的下一个字符，即“要”“有”“直”“升”“机”，即$X$=“想要有直升”，$Y$=“要有直升机”。

现在我们考虑序列“想要有直升机，想要和你飞到宇宙去”，如果时间步数为5，有以下可能的样本和标签：
* $X$：“想要有直升”，$Y$：“要有直升机”
* $X$：“要有直升机”，$Y$：“有直升机，”
* $X$：“有直升机，”，$Y$：“直升机，想”
* ...
* $X$：“要和你飞到”，$Y$：“和你飞到宇”
* $X$：“和你飞到宇”，$Y$：“你飞到宇宙”
* $X$：“你飞到宇宙”，$Y$：“飞到宇宙去”

可以看到，如果序列的长度为$T$，时间步数为$n$，那么一共有$T-n$个合法的样本，但是这些样本有大量的重合，我们通常采用更加高效的采样方式。我们有两种方式对时序数据进行采样，分别是随机采样和相邻采样。

#####  （1）随机采样

如图所示，首先把时间序列（黑色框）划分为长度等于时间步数`num_steps`的分组（蓝色框），如果最后剩余一段长度不足时间步数，就直接忽略掉。每个分组都是个样本。然后从每个分组中选取`batch_size`，每个分组只取一次。

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/7.PNG)

下面的代码每次从数据里随机采样一个小批量。其中批量大小`batch_size`指每个小批量的样本数，`num_steps`为每个样本所包含的时间步数。 在随机采样中，每个样本是原始序列上任意截取的一段序列。相邻的两个随机小批量在原始序列上的位置不一定相毗邻。因此，我们无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。

代码分析：

```python
import torch
import random
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为对于长度为n的序列，X最多只有包含其中的前n - 1个字符
    num_examples = (len(corpus_indices) - 1) // num_steps  # 下取整，得到不重叠情况下的样本个数
    example_indices = [i * num_steps for i in range(num_examples)]  # 每个样本的第一个字符在corpus_indices中的下标
    random.shuffle(example_indices)  # 做随机采样

    def _data(i):
        # 返回从i开始的长为num_steps的序列
        return corpus_indices[i: i + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(0, num_examples, batch_size):
        # 每次选出batch_size个随机样本
        batch_indices = example_indices[i: i + batch_size]  # 当前batch的各个样本的首字符的下标
        X = [_data(j) for j in batch_indices]  # 取出对应的样本
        Y = [_data(j + 1) for j in batch_indices]  # 取出对应的标签
        yield torch.tensor(X, device=device), torch.tensor(Y, device=device)
        
        
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

输入参数：corpus_indices是序列（即上图黑框），device控制返回的批量放在什么设备上。

输出为：（只是一种情况）

```
X:  tensor([[ 6,  7,  8,  9, 10, 11],
        [18, 19, 20, 21, 22, 23]]) 
Y: tensor([[ 7,  8,  9, 10, 11, 12],
        [19, 20, 21, 22, 23, 24]]) 

X:  tensor([[ 0,  1,  2,  3,  4,  5],
        [12, 13, 14, 15, 16, 17]]) 
Y: tensor([[ 1,  2,  3,  4,  5,  6],
        [13, 14, 15, 16, 17, 18]]) 
```

一共包含两个批量（两个X），对应的就是batch_size=2；其中每个批量包含两个样本（两行），每个样本长度是6，对应num_steps=6。对应的标签Y=X+1。

##### （2）相邻采样

在相邻采样中，相邻的两个随机小批量在原始序列上的位置相毗邻。

如图，假设batch_size=3，则先三等分序列（如果后面多出来则舍去）。红色为第一个batch，绿色为第二个batch，蓝色为第三个batch。这里的`num_steps`应该决定的就是每个颜色块的长度，即每个batch的长度。

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/8.PNG)

实现思路就是将序列转化为一个二维数组，然后每个batch转化数组的一列，如下图所示：

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/9.PNG)

代码分析：

```python
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_len = len(corpus_indices) // batch_size * batch_size  # 保留下来的序列的长度
    corpus_indices = corpus_indices[: corpus_len]  # 仅保留前corpus_len个字符
    indices = torch.tensor(corpus_indices, device=device)
    indices = indices.view(batch_size, -1)  # resize成(batch_size, )
    batch_num = (indices.shape[1] - 1) // num_steps  # 求出这个序列能构成多少个批量，-1的原因与前面代码相同，样本不能包含最后一个字符
    for i in range(batch_num):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
        
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
```

输出结果：

```
X:  tensor([[ 0,  1,  2,  3,  4,  5],
        [15, 16, 17, 18, 19, 20]]) 
Y: tensor([[ 1,  2,  3,  4,  5,  6],
        [16, 17, 18, 19, 20, 21]]) 

X:  tensor([[ 6,  7,  8,  9, 10, 11],
        [21, 22, 23, 24, 25, 26]]) 
Y: tensor([[ 7,  8,  9, 10, 11, 12],
        [22, 23, 24, 25, 26, 27]]) 
```

==（注意这里是被resize过后的，所以和最初采样得到的形状并不相同，输出的结果并不能按照上图来看）==

在代码第六行得到的indices为：

```
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
```

在代码第七行得到的indices为：

```
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
```

##  6 循环神经网络

==循环神经网络=多层感知机+隐藏状态==

###  6.1 先修知识

RNN是一类处理序列数据的神经网络。

核心思想：每一时刻重复使用权重矩阵。

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/11.PNG)

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/12.PNG)

RNN不再是一个马尔可夫模型，每一步预测都基于之前的所有信息。

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/13.PNG)

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/14.PNG)

###  6.2 不含隐藏状态的神经网络

考虑一个含单隐藏层的多层感知机。给定样本数为$n$、输入个数（特征数或特征向量维度）为$d$的小批量数据样本$\rm{X} \in R^{n \times d}$。设隐藏层的激活函数为$\phi$，那么隐藏层的输出$H\in R^{n\times h}$为：
$$
H=\phi(XW_{xh}+b_h)
$$
其中隐藏层权重参数$W_{xh}\in R^{d\times h}$，隐藏层偏差参数 $b_h\in R^{1\times h}$，$h$为隐藏单元个数。上式相加的两项形状不同，因此将按照广播机制相加。把隐藏变量$H$作为输出层的输入，且设输出个数为$q$（如分类问题中的类别数），输出层的输出为:
$$
O=HW_{hq}+b_q
$$
其中输出变量$O\times R^{n\times q}$, 输出层权重参数$W_{hq}\in R^{h\times q}$, 输出层偏差参数$b_q \times R^{1\times q}$。如果是分类问题，我们可以使用softmax(O)来计算输出类别的概率分布。

###  6.3 含隐藏状态的循环神经网络

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/15.PNG)

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task1_Task2_Figures/10.PNG)

