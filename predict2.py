import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from scipy.io import loadmat
import argparse

from model_str import Net2
from train import test, LogTransfer


class TestData(data.Dataset):
    def __init__(self, path, transform=None):
        super(TestData, self).__init__()
        self.transform = transform
        self.mats = [
            os.path.join(path, f)
            for f in os.listdir(path) if f.endswith('mat')
        ]

    def __getitem__(self, i):
        mat = self.mats[i]
        datas = loadmat(mat)
        data = datas['data']
        if self.transform is not None:
            data = self.transform(data)
        return data, i

    def __len__(self):
        return len(self.mats)


def main():

    net = Net2(12)
    # 载入数据
    state_dict = torch.load('./model.pth')
    net.load_state_dict(state_dict)
    net.cuda()
    # 读入test数据
    dat = TestData('./preliminary/TEST')
    dataloader = data.DataLoader(dat, batch_size=32, shuffle=False)
    device = torch.device('cuda:0')
    # 进行预测，得到结果并打印
    pred = test(net, dataloader, device, evaluation=False, return_y_true=False)
    _, pred = pred.max(dim=1)
    pred = pred.cpu().numpy().astype('str')
    # 保存
    names = np.array(
        [mat.split('\\')[-1][:-4] for mat in dat.mats], dtype='str')
    df = pd.DataFrame({'name': names, 'pred': pred})
    df.to_csv('./answers.txt', sep=' ', header=False, index=False)


if __name__ == "__main__":
    main()
