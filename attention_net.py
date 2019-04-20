import math
from collections import Iterable

import torch
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
    args:
        input_c, 输入的channels的大小
        output_c, 输出的channels的大小
        hidden_c, list或者是int，表示中间层，或者叫做bottleneck layer
    '''
    def __init__(self, input_c, output_c, hidden_c):
        super(AttentionBlock, self).__init__()
        # 参数的整理和判断
        self.input_c = input_c
        self.hidden_c = hidden_c
        self.output_c = output_c
        # 创建使用的module
        self.attetion_head = nn.Sequential(
            nn.Conv1d(input_c, hidden_c, 101, padding=50),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_c, output_c, 101, padding=50)
        )
        self.feature_head = nn.Sequential(
            nn.Conv1d(input_c, hidden_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_c, output_c, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        atten = self.sigmoid(self.attetion_head(x))
        feature = self.feature_head(x)
        out = atten * feature
        pad_x = self._padding_zero(x, self.output_c)
        out = self.relu(out + pad_x)
        return out

    def _padding_zero(self, inpt, output_c):
        batch, input_c, seq = inpt.size()
        pad_c = output_c - input_c
        pad_tensor = torch.zeros(batch, pad_c, seq, dtype=torch.float).cuda()
        return torch.cat([inpt, pad_tensor], dim=1)


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
        self, input_size, input_c, block_c, pool_k, pool_s, pool_type='max',
        line_h=[]
    ):
        super(AttentionNet, self).__init__()
        self.input_size = input_size
        self.block_c = block_c

        # start block
        self.start_conv = nn.Conv1d(input_c, input_c * 2, 21, 5, padding=10)
        self.relu = nn.ReLU(inplace=True)

        # pooling
        if pool_type == 'max':
            pooling_class = nn.MaxPool1d
        elif pool_type == 'avg':
            pooling_class = nn.AvgPool1d
        self.pooling = pooling_class(
            pool_k, pool_s, padding=int((pool_k-1)/2))

        #
        self.block_list = nn.ModuleList([])
        block_c_pre = [input_c * 2] + list(block_c)[:-1]
        block_c_hid = [int(i / 2) for i in block_c]
        for bcp, bc, bch in zip(
            block_c_pre, block_c, block_c_hid
        ):
            self.block_list.append(AttentionBlock(bcp, bc, bch))

        self.fc_shape = self.fc_in_shape() * block_c[-1]
        print('the fc_in_shape is %d' % self.fc_shape)
        self.fc_list = nn.ModuleList([])
        fc_pre = [self.fc_shape] + list(line_h)
        fc_nex = list(line_h) + [2]
        for l_p, l_n in zip(fc_pre, fc_nex):
            self.fc_list.append(nn.Linear(l_p, l_n))

    def forward(self, x):
        x = self.relu(self.start_conv(x))
        for atten in self.block_list:
            x = self.pooling(atten(x))
        x = x.view(-1, self.fc_shape)
        for i, fc in enumerate(self.fc_list):
            x = fc(x)
            if i != (len(self.fc_list)-1):
                x = self.relu(x)
        return x

    def fc_in_shape(self):
        shape = self.input_size
        return int(shape / (5 * (5 ** len(self.block_c))))
