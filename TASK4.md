#  Task 4

##  1 机器翻译（MT）

###  1.1 数据预处理

数据集格式：

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task3_Task4_Task5_Figures/40.PNG)

处理代码：

```python
def preprocess_raw(text):
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')  # 去掉乱码，\xa0是空格
    out = ''
    for i, char in enumerate(text.lower()):  # 全改成小写
        if char in (',', '!', '.') and i > 0 and text[i-1] != ' ':  #在每个单词和标点之间加上空格
            out += ' '
        out += char
    return out

text = preprocess_raw(raw_text)
print(text[0:1000])
```

==字符在计算机里是以编码的形式存在，我们通常所用的空格是 \x20 ，是在标准ASCII可见字符 0x20~0x7e 范围内。而 \xa0 属于 latin1 （ISO/IEC_8859-1）中的扩展字符集字符，代表不间断空白符nbsp(non-breaking space)，超出gbk编码范围，是需要去除的特殊字符。再数据预处理的过程中，我们首先需要对数据进行清洗。==

结果：

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task3_Task4_Task5_Figures/41.PNG)

###  1.2 分词

字符串->单词组成的列表

```python
num_examples = 50000
source, target = [], []
for i, line in enumerate(text.split('\n')):  # 通过换行符来分开每一行
    if i > num_examples:
        break
    parts = line.split('\t')  # 通过tab符分开三项，就可以取前两项，如第一行，就是go. , va ! , cc-by.....................................
    if len(parts) >= 2:
        source.append(parts[0].split(' '))  #用空格把每个单词区分开
        target.append(parts[1].split(' '))
        
source[0:3], target[0:3]
```

结果：

```
([['go', '.'], ['hi', '.'], ['hi', '.']],
 [['va', '!'], ['salut', '!'], ['salut', '.']])
```

第一行为英语单词，第二行为法语单词。

###  1.3 建立词典

为英语和法语各建立词典，收录各个语言中出现过的单词

```python
def build_vocab(tokens):
    tokens = [token for line in tokens for token in line]  #取出所有单词，连成单词列表
    return d2l.data.base.Vocab(tokens, min_freq=3, use_special_tokens=True) # 调用vacab类

src_vocab = build_vocab(source)
len(src_vocab)
```

其中Vocab定义如下：

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task3_Task4_Task5_Figures/42.PNG)

###  1.4 读取数据集

```python
def pad(line, max_len, padding_token):
    if len(line) > max_len:  #大于的话就截去
        return line[:max_len]
    return line + [padding_token] * (max_len - len(line)) #否则补足
pad(src_vocab[source[0]], 10, src_vocab.pad)
```

输出：

```
[38, 4, 0, 0, 0, 0, 0, 0, 0, 0]
```

```python
def build_array(lines, vocab, max_len, is_source):
    lines = [vocab[line] for line in lines]
    if not is_source:
        lines = [[vocab.bos] + line + [vocab.eos] for line in lines] #加bos和eos符号，判断开始和结尾
    array = torch.tensor([pad(line, max_len, vocab.pad) for line in lines])
    valid_len = (array != vocab.pad).sum(1) #设置有效长度，用来保存句子原本的长度，第一个维度
    return array, valid_len  #得到了id组成的tensor和valid_len
```

```python
def load_data_nmt(batch_size, max_len): # This function is saved in d2l.
    src_vocab, tgt_vocab = build_vocab(source), build_vocab(target)  #生成词典
    src_array, src_valid_len = build_array(source, src_vocab, max_len, True)  #生成英语的
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, max_len, False)  #法语的
    train_data = data.TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)  #判断这四个内容是不是一一对应的
    train_iter = data.DataLoader(train_data, batch_size, shuffle=True)  # 数据生成器
    return src_vocab, tgt_vocab, train_iter
```

```python
src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size=2, max_len=8)
for X, X_valid_len, Y, Y_valid_len, in train_iter:
    print('X =', X.type(torch.int32), '\nValid lengths for X =', X_valid_len,
        '\nY =', Y.type(torch.int32), '\nValid lengths for Y =', Y_valid_len) #每次只生成一组
    break
```

输出结果：

```
X = tensor([[   5,   24,    3,    4,    0,    0,    0,    0],
        [  12, 1388,    7,    3,    4,    0,    0,    0]], dtype=torch.int32) 
Valid lengths for X = tensor([4, 5]) 
Y = tensor([[   1,   23,   46,    3,    3,    4,    2,    0],
        [   1,   15,  137,   27, 4736,    4,    2,    0]], dtype=torch.int32) 
Valid lengths for Y = tensor([7, 7])
```

###  1.5 Encoder-Decoder

为了解决输入和输出不对等

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task3_Task4_Task5_Figures/43.PNG)

先把输入变成隐藏状态，当Decoder最后一个得到eos时，认为结束了。

###  1.6 Sequence to Sequence模型

Encoder可以是循环神经网络，然后输出Hidden state

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task3_Task4_Task5_Figures/44.PNG)

![](https://github.com/cannibalistic-galaxy/Deeplearning-with-pytorch/blob/master/Task3_Task4_Task5_Figures/45.PNG)

```python
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size,num_hiddens, num_layers, dropout=dropout)
   
    def begin_state(self, batch_size, device):
        return [torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device),
                torch.zeros(size=(self.num_layers, batch_size, self.num_hiddens),  device=device)]
    def forward(self, X, *args):
        X = self.embedding(X) # X shape: (batch_size, seq_len, embed_size)
        X = X.transpose(0, 1)  # RNN needs first axes to be time
        # state = self.begin_state(X.shape[1], device=X.device)
        out, state = self.rnn(X)  # 循环神经网络输入必须是一个时序
        # The shape of out is (seq_len, batch_size, num_hiddens).
        # state contains the hidden state and the memory cell
        # of the last time step, the shape is (num_layers, batch_size, num_hiddens)
        return out, state
```

