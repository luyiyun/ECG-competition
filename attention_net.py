import math
from collections import Iterable

import torch.nn as nn


def shape_calculate(i_s, k, p, s, d):
    ''' 用于计算一维卷积、pooling的输出大小 '''
    return math.floor((i_s + 2 * p - d * (k - 1) - 1) / s + 1)


def conv_same(input_c, output_c, k):
    ''' 自动计算padding使得经过卷积的seq的长度不变  '''
    return nn.Conv1d(input_c, output_c, k, stride=1, padding=int((k - 1)/2))


def int_or_iterable(x, length=None):
    ''' 对hidden_c的类型做一下判断，代码更加robust '''
    if isinstance(x, int):
        x = [x]
        if length is None:
            return x
        else:
            return x * length
    elif isinstance(x, Iterable):
        x = list(x)
        return x
    else:
        raise ValueError(
            'the type of x: %s is not fit.' % str(type(x)))


class AttentionBlock(nn.Module):
    '''
    一个attention block，输入和输出的seq长度是一样的
    这里使用的卷积都是stride=1和自动padding使得得到的seq的长度是一样的
    在使用attention p（由softmax生成）时，先将输入使用1x1卷积进行升维，
    再进行attention mask和residual link，这样整个attention block也可以
    增加channels
    args:
        input_c, 输入的channels的大小
        output_c, 输出的channels的大小
        hidden_c, list或者是int，表示中间层，或者叫做bottleneck layer
        conv_k，使用的卷积核的大小
    '''
    def __init__(self, input_c, output_c, hidden_c, conv_k):
        super(AttentionBlock, self).__init__()
        # 参数的整理和判断
        self.input_c = input_c
        self.hidden_c = int_or_iterable(hidden_c)
        self.conv_k = int_or_iterable(conv_k, length=len(self.hidden_c))
        # 创建使用的module
        hidden_c_pre = [input_c] + self.hidden_c[:-1]
        self.conv_list = nn.ModuleList([])
        self.bn_list = nn.ModuleList([])
        for k, c_p, c in zip(self.conv_k, hidden_c_pre, self.hidden_c):
            self.conv_list.append(conv_same(c_p, c, k))
            self.bn_list.append(nn.BatchNorm1d(c))
        self.relu = nn.ReLU()
        # 1x1卷积来生成softmax分数，进行self-attention
        self.conv_attention = nn.Conv1d(self.hidden_c[-1], 1, 1)
        self.softmax = nn.Softmax(dim=2)
        # 1x1卷积来对原始的x增加channels
        self.conv_addc = nn.Conv1d(input_c, output_c, 1)

    def forward(self, x):
        out = x
        for conv, bn in zip(self.conv_list, self.bn_list):
            out = self.relu(bn(conv(out)))
        out = self.softmax(self.conv_attention(out))
        x_ = self.conv_addc(x)
        out = out * x_
        out = out + x_
        return out


class AttentionNet(nn.Module):
    '''
    多个attention block堆叠而成的Attention Net，
    其中首先进行一次较大的conv+pooling，得到的channels是第一个block_c，
    之后依据block_c和block_k，pool_k和pool_s堆叠attention block和max pooling，
        attention block不会改变seq的长度，但会改变channels的数量，
        pooling不会改变channels的数量，但会减小seq的长度
    args：
        input_size：输入的seq的长度
        input_c：输入的channels的数量
        block_c：每个block输出的channels的数量
        block_k：每个block使用的卷积核的大小
        pool_k：使用的pool的kernel的大小
        pool_s：使用的pool的stride的大小
        pool_type：使用的pool的类型
        line_h：后面的fc的hidden units
    '''
    def __init__(
        self, input_size, input_c, block_c, block_k, pool_k, pool_s,
        pool_type='max', line_h=[]
    ):
        super(AttentionNet, self).__init__()
        self.input_size = input_size
        self.pool_k = int_or_iterable(pool_k)
        self.pool_s = int_or_iterable(pool_s, length=len(self.pool_k))

        # 创建多个Attention block，其中bottleneck layer是两层，其channels是
        #   输入的1/2
        self.block_list = nn.ModuleList([])
        block_c_pre = [input_c] + list(block_c)[:-1]
        block_c_hid = [int(i / 2) for i in block_c]
        for bcp, bc, bch, bk in zip(
            block_c_pre, block_c, block_c_hid, block_k
        ):
            self.block_list.append(AttentionBlock(bcp, bc, [bch] * 2, bk))
        # 创建多个pooling layer
        self.pool_list = nn.ModuleList([])
        if pool_type == 'max':
            pool_class = nn.MaxPool1d
        elif pool_type == 'avg':
            pool_class = nn.AvgPool1d
        for p_k, p_s in zip(self.pool_k, self.pool_s):
            self.pool_list.append(pool_class(p_k, stride=p_s))
        # 创建最后的分类fc
        self.fc_shape = self.fc_in_shape() * block_c[-1]
        print('the fc_in_shape is %d' % self.fc_shape)
        self.fc_list = nn.ModuleList([])
        fc_pre = [self.fc_shape] + list(line_h)
        fc_nex = list(line_h) + [2]
        for l_p, l_n in zip(fc_pre, fc_nex):
            self.fc_list.append(nn.Linear(l_p, l_n))

    def forward(self, x):
        for atten, pool in zip(self.block_list, self.pool_list):
            x = pool(atten(x))
        x = x.view(-1, self.fc_shape)
        for i, fc in enumerate(self.fc_list):
            x = fc(x)
            if i != (len(self.fc_list)-1):
                x = self.relu(x)
        return x

    def fc_in_shape(self):
        shape = self.input_size
        for p_k, p_s in zip(self.pool_k, self.pool_s):
            shape = shape_calculate(shape, k=p_k, p=0, s=p_s, d=1)
            if shape < 1:
                raise ValueError('The shape is %d, smaller than 1' % shape)
        return shape
