import math
from itertools import chain

import torch.nn as nn
import torch.nn.functional as F


def interlace(*iters):
    res = []
    for js in zip(*iters):
        for j in js:
            res.append(j)
    return res


class CnnNet(nn.Module):
    '''
    Conv1d-net
    '''
    def __init__(
        self, input_c, input_s=5000, conv_c=(6, 6, 6), conv_k=(100, 6, 6),
        conv_s=(1, 1, 1), conv_p=(0, 0, 0), pool_k=20, pool_s=5,
        pool_type='max', bn=True, line_h=[]
    ):
        super(CnnNet, self).__init__()
        self.input_c = input_c
        self.input_s = input_s
        self.conv_c = conv_c
        self.conv_k = conv_k
        self.conv_s = conv_s
        self.conv_p = conv_p
        self.pool_k = [pool_k] * len(conv_k) \
            if isinstance(pool_k, int) else pool_k
        self.pool_s = [pool_s] * len(conv_s) \
            if isinstance(pool_s, int) else pool_s
        self.pool_p = [0] * len(conv_p)
        self.bn = bn
        convs = []
        conv_c_pre = [input_c] + list(conv_c)[:-1]
        for c_p, c_n, k, s, p in zip(
            conv_c_pre, conv_c, conv_k, conv_s, conv_p
        ):
            convs.append(nn.Conv1d(c_p, c_n, k, stride=s, padding=p))
        self.convs = nn.ModuleList(convs)
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(pool_k, stride=pool_s)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool1d(pool_k, stride=pool_s)
        self.in_features = self.fc_in_shape() * conv_c[-1]

        line_pre = [self.in_features] + line_h
        line_nex = line_h + [2]
        self.fcs = nn.ModuleList(
            [nn.Linear(lp, ln) for lp, ln in zip(line_pre, line_nex)])

        if bn:
            bns = []
            for c in conv_c:
                bns.append(nn.BatchNorm1d(c))
            self.bns = nn.ModuleList(bns)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = self.pool(F.relu(bn(conv(x))))
        for i, fc in enumerate(self.fcs):
            if i == 0:
                x = x.view(-1, self.in_features)
            x = fc(x)
            if i != (len(self.fcs)-1):
                x = F.relu(x)
        return x

    def fc_in_shape(self):
        # all_k = list(chain(self.conv_k, self.pool_k))
        # all_s = list(chain(self.conv_s, self.pool_s))
        # all_p = list(chain(self.conv_p, self.pool_p))
        all_k = interlace(self.conv_k, self.pool_k)
        all_s = interlace(self.conv_s, self.pool_s)
        all_p = interlace(self.conv_p, self.pool_p)
        shape = self.input_s
        for k, s, p in zip(all_k, all_s, all_p):
            shape = self._one_shape(shape, k, p, s)
            if shape < 1:
                raise ValueError('某个feature map的维度太小了')
        return shape

    @staticmethod
    def _one_shape(i_s, k, p, s):
        return math.floor((i_s + 2 * p - (k - 1) - 1) / s + 1)
