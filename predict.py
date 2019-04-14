import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from scipy.io import loadmat
import argparse

from simple_net import CnnNet
from attention_net import AttentionNet
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-mp', '--model_path',
        default='E:/subject-other/心电比赛/results/fc1',
        help='模型、结果和配置保存的文件夹')
    parser.add_argument(
        '-o', '--output', default='./answers.txt',
        help='保存的结果文件名，默认是answers.txt')
    parser.add_argument(
        '-dd', '--data_dir', default='E:/subject-other/心电比赛/TEST',
        help='Test数据所在的dir'
    )
    args = parser.parse_args()
    # 根据config来重现net
    config = torch.load(os.path.join(args.model_path, 'config.pth'))
    if config['mode'] == 'dense':
        net = CnnNet(
            input_c=config['input_size'], conv_c=config['conv_c'],
            conv_k=config['conv_k'], conv_s=config['conv_s'],
            conv_p=config['conv_p'], pool_k=config['pool_k'],
            pool_s=config['pool_s'], pool_type=config['pool_type'],
            line_h=config['linear_hidden']
        )
    elif config['mode'] == 'attention':
        net = AttentionNet(
            input_size=5000, input_c=config['input_size'],
            block_c=config['conv_c'], block_k=config['conv_k'],
            pool_k=config['pool_k'], pool_s=config['pool_s'],
            pool_type=config['pool_type'], line_h=config['linear_hidden']
        )
    # 载入数据
    state_dict = torch.load(os.path.join(args.model_path, 'model.pth'))
    net.load_state_dict(state_dict)
    net.cuda()
    # 读入test数据
    if config['log_transfer']:
        transfer = LogTransfer()
    else:
        transfer = None
    dat = TestData(args.data_dir, transfer)
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
    df.to_csv(args.output, sep=' ', header=False, index=False)


if __name__ == "__main__":
    main()
