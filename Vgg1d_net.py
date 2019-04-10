import os
import copy
import math
from itertools import chain
import json

import numpy as np
import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import progressbar as pb
import argparse

from cnn1d_net import MyDataSet, train


class Vgg1dNet(nn.Module):
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
        self.pool_k = [pool_k] * len(conv_k)
        self.pool_s = [pool_s] * len(conv_s)
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
        all_k = list(chain.from_iterable(zip(self.conv_k, self.pool_k)))
        all_s = list(chain.from_iterable(zip(self.conv_s, self.pool_s)))
        all_p = list(chain.from_iterable(zip(self.conv_p, self.pool_p)))
        shape = self.input_s
        for k, s, p in zip(all_k, all_s, all_p):
            shape = self._one_shape(shape, k, p, s)
            if shape < 1:
                raise ValueError('ĳ��feature map��ά��̫С��')
        return shape

    @staticmethod
    def _one_shape(i_s, k, p, s):
        return math.floor((i_s + 2 * p - (k - 1) - 1) / s + 1)


def train(
    net, criterion, dataloaders, optimizer, device, epochs=50,
    early_stop=50
):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=5, min_lr=1e-5, verbose=True)
    best_acc = 0.
    best_auc = 0.
    best_model_wts = copy.deepcopy(net.state_dict())
    no_improve = 0
    history = {
        'loss': [], 'acc': [], 'auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []}
    for epoch in range(epochs):
        for phase in ["train", "val"]:
            # ÿ��batch��Ҫ��¼�Ľ����һ����ѭ��������epoch�����metrics
            running_loss = 0.0
            running_corrects = 0
            predicts = []
            y_true = []
            # ����train��loss�в�ͬ�Ĵ���train�н�����
            if phase == "train":
                net.train()
                if epoch == 0:
                    scheduler.step(500)
                else:
                    scheduler.step(history['val_loss'][-1])
                widges = [
                    'epoch: %d' % epoch, '| ', pb.Counter(),
                    pb.Bar(), pb.AdaptiveETA()]
                iterator = pb.progressbar(dataloaders[phase], widgets=widges)
            else:
                net.eval()
                iterator = dataloaders[phase]
            # ǰ�򴫲�������loss��������train���У������ݶȣ����򴫲�����
            #   ��no grad��ģʽ���ۼ�loss��correct���������Ա����loss��acc
            #   ����¼predict��y_true���Ա����auc
            for inputs, labels in iterator:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels)
                    predicts.append(outputs)
                    y_true.append(labels)
            # ÿ��epoch����������metrics����������history��
            with torch.no_grad():
                predicts = torch.cat(predicts, 0)
                y_true = torch.cat(y_true, 0)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(
                    dataloaders[phase].dataset)
                epoch_auc = roc_auc_score(
                    y_true.cpu().numpy(), predicts.cpu().numpy()[:, 1])
                if phase == 'train':
                    history['loss'].append(epoch_loss)
                    history['acc'].append(epoch_acc)
                    history['auc'].append(epoch_auc)
                elif phase == 'val':
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)
                    history['val_auc'].append(epoch_auc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_auc = epoch_auc
                        best_model_wts = copy.deepcopy(net.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1
                print(
                    "%s, Loss: %.4f ACC: %.4f, AUC: %.4f" % (
                        phase, epoch_loss, epoch_acc, epoch_auc))
        # ��ǰֹͣ
        if no_improve == early_stop:
            print('Early Stop...')
            break
    net.load_state_dict(best_model_wts)
    print('valid, best_acc: %.4f, best_auc: %.4f' % (best_acc, best_auc))
    # �����test���ݼ����������test��Ԥ��
    if 'test' in dataloaders.keys():
        print('Testing...')
        with torch.no_grad():
            predicts = []
            y_true = []
            running_loss = 0.
            running_corrects = 0
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device)
                outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                predicts.append(outputs)
                y_true.append(labels)
            predicts = torch.cat(predicts, 0)
            y_true = torch.cat(y_true, 0)
            epoch_loss = running_loss / len(dataloaders['test'].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders['test'].dataset)
            epoch_auc = roc_auc_score(
                y_true.cpu().numpy(), predicts.cpu().numpy()[:, 1])
    test_results = [epoch_loss, epoch_acc.item(), epoch_auc]
    print(test_results)

    return test_results, history, best_model_wts


def main():
    # ���������в����������н���
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cc', '--conv_c', default=[20, 40, 60, 80], type=int, nargs='+')
    parser.add_argument(
        '-ck', '--conv_k', default=[10] * 4, type=int, nargs='+')
    parser.add_argument(
        '-cs', '--conv_s', default=[1] * 4, type=int, nargs='+')
    parser.add_argument(
        '-cp', '--conv_p', default=[0] * 4, type=int, nargs='+')
    parser.add_argument(
        '-lh', '--linear_hidden', default=[], type=int, nargs='+')
    parser.add_argument('-pk', '--pool_k', default=20, type=int)
    parser.add_argument('-ps', '--pool_s', default=5, type=int)
    parser.add_argument('-pt', '--pool_type', default='max')
    parser.add_argument('-rs', '--random_seed', default=1234, type=int)
    parser.add_argument('-rd', '--root_dir', default='./TRAIN')
    parser.add_argument('-lr', '--learning_rate', default=.01, type=float)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epoch', default=500, type=int)
    parser.add_argument('-is', '--input_size', default=12, type=int)
    parser.add_argument('-sd', '--save_dir', default='./results/exam')
    args = parser.parse_args()
    device = torch.device("cuda:0")

    # ��������
    txt_file = np.loadtxt("./reference.txt", dtype="str")
    train_txt, valid_txt = train_test_split(
        txt_file, test_size=0.2, shuffle=True,
        random_state=args.random_seed, stratify=txt_file[:, 1])
    valid_txt, test_txt = train_test_split(
        valid_txt, test_size=0.5, shuffle=True,
        random_state=args.random_seed, stratify=valid_txt[:, 1])
    datasets = {
        "train": MyDataSet(train_txt, args.root_dir),
        "val": MyDataSet(valid_txt, args.root_dir),
        "test": MyDataSet(test_txt, args.root_dir)
    }
    dataloaders = {
        k: DataLoader(v, batch_size=args.batch_size)
        for k, v in datasets.items()}

    # ��������
    net = CnnNet(
        input_c=args.input_size, conv_c=args.conv_c,
        conv_k=args.conv_k, conv_s=args.conv_s,
        conv_p=args.conv_p, pool_k=args.pool_k, pool_s=args.pool_s,
        pool_type=args.pool_type, line_h=args.linear_hidden
    )
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9)

    # ѵ������
    test_result, hist, state_dict = train(
        net, criterion, dataloaders, optimizer, device, epochs=args.epoch)

    # ����ģ�ͺͽ��
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    torch.save(state_dict, os.path.join(args.save_dir, 'model.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(args.save_dir, 'train.csv'))
    np.savetxt(os.path.join(args.save_dir, 'test.txt'), test_result)
    torch.save(args.__dir__, os.path.join(args.save_dir, 'config.json'))
    # with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
    #     json.dump(args.__dir__, f)


if __name__ == '__main__':
    main()
