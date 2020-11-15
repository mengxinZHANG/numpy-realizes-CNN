import numpy as np

class Dropout():
    def __init__(self, drop_rate=0.5, is_train=True):
        """
        :param drop_rate: 随机丢弃神经元的概率
        :param is_train: 当前是否为训练状态
        """
        self.drop_rate = drop_rate
        self.is_train = is_train
        self.fix_value = 1 - drop_rate   # 修正期望，保证输出值的期望不变


    def forward(self, x):
        """
        :param x:[N, m] N为batch_size, m为神经元个数
        :return:
        """
        if self.is_train==False:    # 当前为测试状态
            return x
        else:             # 当前为训练状态
            N, m = x.shape
            self.save_mask = np.random.uniform(0, 1, m) > self.drop_rate   # save_mask中为保留的神经元
            return (x * self.save_mask) / self.fix_value


    def backward(self, eta):
        if self.is_train==False:
            return eta
        else:
            return eta * self.save_mask


