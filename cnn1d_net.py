import os
import copy
import math
from itertools import chain

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

from attention_net import AttentionNet


class MyDataSet(Dataset):
    '''
    pytorch  dataset类，
    args:
        text_file: ndarray, N x 2, N是样本数，
            第一列是每个人数据文件的名字，第二列是每个人的标签。
        trainval_root：所有数据文件所在的文件夹路径。
    '''

    def __init__(self, txt_file, trainval_root):
        super(MyDataSet, self).__init__()
        self.txt_file = txt_file
        self.trainval_root = trainval_root

    def __len__(self):
        return len(self.txt_file)

    def __getitem__(self, idx):
        file_name = self.txt_file[idx, 0]
        label = self.txt_file[idx, 1]
        # 将路径和文件名merge在一起
        file_path = os.path.join(self.trainval_root, file_name + ".mat")
        file = io.loadmat(file_path)
        return file["data"], np.int(label)


class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()
        self.rnn1 = nn.GRU(input_size, 50, 1, dropout=0.5)
        self.rnn2 = nn.GRU(50, 1, 1, dropout=0.5)
        self.fc = nn.Linear(5000, 2)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc(x.permute(1, 0, 2).view(-1, 5000))
        # x = self.fc(x[-1])
        return x


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
        all_k = list(chain(self.conv_k, self.pool_k))
        all_s = list(chain(self.conv_s, self.pool_s))
        all_p = list(chain(self.conv_p, self.pool_p))
        shape = self.input_s
        for k, s, p in zip(all_k, all_s, all_p):
            shape = self._one_shape(shape, k, p, s)
            if shape < 1:
                raise ValueError('某个feature map的维度太小了')
        return shape

    @staticmethod
    def _one_shape(i_s, k, p, s):
        return math.floor((i_s + 2 * p - (k - 1) - 1) / s + 1)


def test(net, criterion, dataloader, device, evaluation=True):
    ''' 根据训练好的net来进行预测，可以输出loss和评价指标 '''
    print('Testing...')
    with torch.no_grad():
        predicts = []
        y_true = []
        running_loss = 0.
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            if evaluation:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            predicts.append(outputs)
            if evaluation:
                y_true.append(labels)
        predicts = torch.cat(predicts, 0)
        if evaluation:
            y_true = torch.cat(y_true, 0)
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_auc = roc_auc_score(
                y_true.cpu().numpy(), predicts.cpu().numpy()[:, 1])
            test_results = [epoch_loss, epoch_acc.item(), epoch_auc]
            return predicts, test_results
        else:
            return predicts


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
            # 每个batch需要记录的结果，一遍在循环完整个epoch后计算metrics
            running_loss = 0.0
            running_corrects = 0
            predicts = []
            y_true = []
            # 对于train和loss有不同的处理，train有进度条
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
            # 前向传播，计算loss，（对于train还有：计算梯度，反向传播），
            #   在no grad的模式下累加loss和correct的数量，以便计算loss和acc
            #   并记录predict和y_true，以便计算auc
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
            # 每个epoch结束，计算metrics，并保存在history中
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
        # 提前停止
        if no_improve == early_stop:
            print('Early Stop...')
            break
    net.load_state_dict(best_model_wts)
    print('valid, best_acc: %.4f, best_auc: %.4f' % (best_acc, best_auc))
    # 如果有test数据集，则进行他test的预测
    if 'test' in dataloaders.keys():
        _, test_results = test(
            net, criterion, dataloaders['test'], device, evaluation=True)
        print(test_results)

    return test_results, history, best_model_wts


def main():
    # 载入命令行参数，并进行解析
    parser = argparse.ArgumentParser()
    parser.add_argument('save', default='exam', nargs='?',
                        help='保存结果的文件夹的名称')
    parser.add_argument(
        '-cc', '--conv_c', default=[20, 40, 60, 80], type=int, nargs='+',
        help='卷积核层输入的feature map的channels， 默认是[20, 40, 60, 80]'
    )
    parser.add_argument(
        '-ck', '--conv_k', default=[10] * 4, type=int, nargs='+',
        help=(
            (
                '使用的卷积的卷积核大小，如果是使用attention，则是'
                'attention block使用的卷积核大小，默认是10'
            )
        )
    )
    parser.add_argument(
        '-cs', '--conv_s', default=[1] * 4, type=int, nargs='+',
        help='使用的卷积操作的stride，默认是1'
    )
    parser.add_argument(
        '-cp', '--conv_p', default=[0] * 4, type=int, nargs='+',
        help='使用的卷积操作的padding，默认是0'
    )
    parser.add_argument(
        '-lh', '--linear_hidden', default=[], type=int, nargs='+',
        help='最后要接的fc的隐层的units，默认是空，即直接进行softmax'
    )
    parser.add_argument(
        '-pk', '--pool_k', default=20, type=int, nargs='+',
        help=(
            '每个conv后跟的pooling的kernel的大小，默认是20'
            '当使用attention时，其指代的是每个block后跟的那个pooling layer'
        ))
    parser.add_argument(
        '-ps', '--pool_s', default=5, type=int, nargs='+',
        help=(
            'pooling的stride，默认是5'
            '当使用attention时，其指代的是每个block后跟的那个pooling layer'
        ))
    parser.add_argument(
        '-pt', '--pool_type', default='max',
        help='pooling的种类，默认是max，还可以是avg')
    parser.add_argument(
        '-rs', '--random_seed', default=1234, type=int, help='随机种子')
    parser.add_argument(
        '-tvr', '--trainval_root', default='E:/subject-other/心电比赛/TRAIN',
        help='训练集所在的路径，默认是E:/subject-other/心电比赛/TRAIN')
    parser.add_argument(
        '-lr', '--learning_rate', default=.01, type=float,
        help='默认的初始学习率是0.01'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=32, type=int,
        help='默认的batch size是32')
    parser.add_argument(
        '-e', '--epoch', default=500, type=int,
        help='最大的epoch数量，默认是500')
    parser.add_argument(
        '-is', '--input_size', default=12, type=int,
        help='输入的特征的大小，默认是12')
    parser.add_argument(
        '-sr', '--save_root', default='E:/subject-other/心电比赛/results',
        help='保存结果的根目录，默认是E:/subject-other/心电比赛/results')
    parser.add_argument(
        '-m', '--mode', default='dense',
        help='使用的网络类型，默认是dense，可以是attention')
    args = parser.parse_args()
    device = torch.device("cuda:0")

    # 导入数据
    txt_file = np.loadtxt("./reference.txt", dtype="str")
    train_txt, valid_txt = train_test_split(
        txt_file, test_size=0.2, shuffle=True,
        random_state=args.random_seed, stratify=txt_file[:, 1])
    valid_txt, test_txt = train_test_split(
        valid_txt, test_size=0.5, shuffle=True,
        random_state=args.random_seed, stratify=valid_txt[:, 1])
    datasets = {
        "train": MyDataSet(train_txt, args.trainval_root),
        "val": MyDataSet(valid_txt, args.trainval_root),
        "test": MyDataSet(test_txt, args.trainval_root)
    }
    dataloaders = {
        k: DataLoader(v, batch_size=args.batch_size)
        for k, v in datasets.items()}

    # 构建网络
    if args.mode == 'dense':
        net = CnnNet(
            input_c=args.input_size, conv_c=args.conv_c,
            conv_k=args.conv_k, conv_s=args.conv_s,
            conv_p=args.conv_p, pool_k=args.pool_k, pool_s=args.pool_s,
            pool_type=args.pool_type, line_h=args.linear_hidden
        )
    elif args.mode == 'attention':
        net = AttentionNet(
            input_size=5000, input_c=args.input_size,
            block_c=args.conv_c, block_k=args.conv_k, pool_k=args.pool_k,
            pool_s=args.pool_s, pool_type=args.pool_type,
            line_h=args.linear_hidden
        )
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9)

    # 训练网络
    test_result, hist, state_dict = train(
        net, criterion, dataloaders, optimizer, device, epochs=args.epoch)

    # 保存模型和结果
    save_dir = os.path.join(args.save_root, args.save)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(state_dict, os.path.join(save_dir, 'model.pth'))
    pd.DataFrame(hist).to_csv(os.path.join(save_dir, 'train.csv'))
    np.savetxt(os.path.join(save_dir, 'test.txt'), test_result)
    torch.save(args.__dict__, os.path.join(save_dir, 'config.pth'))


if __name__ == '__main__':
    main()
