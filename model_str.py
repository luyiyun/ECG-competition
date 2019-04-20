import torch.nn as nn


class Net2(nn.Module):
    """
    模型2
    """
    def __init__(self, in_channels, num_classes=2):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 20, 20)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(10, 5)
        self.conv2 = nn.Conv1d(20, 20, 10)
        self.conv3 = nn.Conv1d(20, 20, 10)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(20)
        self.globalpool = nn.AdaptiveAvgPool1d(1)
        self.fc_layer1 = nn.Linear(20, 40)
        self.fc_layer2 = nn.Linear(40, num_classes)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.bn3(self.conv3(x))))
        x = self.globalpool(x)
        x = x.squeeze()
        x = self.fc_layer1(x)
        x = self.dropout(x)
        x = self.fc_layer2(x)
        return x
