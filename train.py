import os
import copy

import numpy as np
import pandas as pd
import scipy.io as io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import progressbar as pb
import argparse
from torchvision import transforms

from attention_net import AttentionNet
from simple_net import CnnNet


class MyDataSet(Dataset):
    '''
    pytorch  dataset类，
    args:
        text_file: ndarray, N x 2, N是样本数，
            第一列是每个人数据文件的名字，第二列是每个人的标签。
        trainval_root：所有数据文件所在的文件夹路径。
    '''

    def __init__(self, txt_file, trainval_root, transfrom=None):
        super(MyDataSet, self).__init__()
        self.txt_file = txt_file
        self.trainval_root = trainval_root
        self.transfrom = transfrom

    def __len__(self):
        return len(self.txt_file)

    def __getitem__(self, idx):
        file_name = self.txt_file[idx, 0]
        label = self.txt_file[idx, 1]
        # 将路径和文件名merge在一起
        file_path = os.path.join(self.trainval_root, file_name + ".mat")
        file = io.loadmat(file_path)
        data = file['data']
        if self.transfrom is not None:
            data = self.transfrom(data)
        return data, np.int(label)


class LogTransfer:

    def __init__(self, zero_one=True, epsilon=10e-5):
        self.epsilon = epsilon
        self.zero_one = zero_one

    def __call__(self, x):
        if self.zero_one:
            xmin = x.min(axis=1, keepdims=True)
            xmax = x.max(axis=1, keepdims=True)
            x = (x - xmin) / (xmax - xmin)
        return np.log(x + self.epsilon)


def test(
    net, dataloader, device, evaluation=True, criterion=None,
    return_y_true=False
):
    ''' 根据训练好的net来进行预测，可以输出loss和评价指标 '''
    print('Testing...')
    if evaluation and criterion is None:
        raise ValueError('if evaluation is True, criterion must be given.')
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
            if evaluation or return_y_true:
                y_true.append(labels)
        predicts = torch.cat(predicts, 0)
        if evaluation:
            y_true = torch.cat(y_true, 0)
            y_true = y_true.cpu().numpy()
            positive = predicts.cpu().numpy()[:, 1]
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_auc = roc_auc_score(y_true, positive)
            epoch_f1 = f1_score(y_true, positive > 0.5)
            test_results = [epoch_loss, epoch_acc.item(), epoch_auc, epoch_f1]
            if return_y_true:
                return predicts, y_true, test_results
            else:
                return predicts, test_results
        elif return_y_true:
            return predicts, y_true
        else:
            return predicts


def train(
    net, criterion, dataloaders, optimizer, device, epochs=50,
    early_stop=50, scheduler=None
):
    best_loss = 100.
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
                epoch_acc = (running_corrects.double() / len(
                    dataloaders[phase].dataset)
                ).item()
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
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
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
            net, dataloaders['test'], device, evaluation=True,
            criterion=criterion)
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
        '-rs', '--random_seed', default=2345, type=int, help='随机种子')
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
    parser.add_argument(
        '-lr_s', '--lr_scheduler', default='ROP',
        help='学习率下降的方式，ROP or Step')
    parser.add_argument('-opt', '--optimizer',
                        default='sgd', help='优化器，sgd or adam')
    parser.add_argument(
        '-lt', '--log_transfer', action='store_true',
        help='使用此参数即代表使用log转换')
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
    if args.log_transfer:
        transfer = LogTransfer()
    else:
        transfer = None
    datasets = {
        "train": MyDataSet(train_txt, args.trainval_root, transfrom=transfer),
        "val": MyDataSet(valid_txt, args.trainval_root, transfrom=transfer),
        "test": MyDataSet(test_txt, args.trainval_root, transfrom=transfer)
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
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            net.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            net.parameters(), lr=args.learning_rate)
    if args.lr_scheduler == 'ROP':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.7, patience=5, min_lr=1e-5, verbose=True)
    elif args.lr_scheduler == 'Step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.9)

    # 训练网络
    test_result, hist, state_dict = train(
        net, criterion, dataloaders, optimizer,
        device, epochs=args.epoch, scheduler=scheduler)

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
