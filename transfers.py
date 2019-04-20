import numpy as np
import torch


class DropoutBrust:

    def __init__(self, p=0.1, interval=50, size=5000):
        super(DropoutBrust, self).__init__()
        self.p = p
        self.interval = interval
        self.size = size
        self.point_num = int(p * size)
        self.half = int(interval / 2)

    def __call__(self, x):
        point_index = np.random.randint(self.size, size=self.point_num)
        for i in point_index:
            x[:, (i - self.half): (i + self.half)] = 0.
        return x

