import numpy as np

class softmax():
    # def __init__(self, shape):
    #     """
    #     :param shape:[N, m] 其中N为batch_size，m为要预测的类别
    #     """
    #     self.out = np.zeros(shape)
    #     self.eta = np.zeros(shape)

    def calculate_loss(self, x, label):
        """
        :param x: 上一层输出的向量：[N, m] 其中N表示batch，m表示输出节点个数
        :param label:数据的真实标签：[N]
        :return:
        """
        N, _ = x.shape
        self.label = np.zeros_like(x)
        for i in range(self.label.shape[0]):
            self.label[i, label[i]] = 1

        self.x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])   # 为了防止x中出现极值导致溢出，每个样本减去其中最大的值
        sum_x = np.sum(self.x, axis=1)[:, np.newaxis]
        self.prediction = self.x / sum_x

        self.loss = -np.sum(np.log(self.prediction+1e-6) * self.label)  # 防止出现log(0)的情况
        return self.loss / N

    def prediction_func(self, x):
        x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])  # 为了防止x中出现极值导致溢出，每个样本减去其中最大的值
        sum_x = np.sum(x, axis=1)[:, np.newaxis]
        self.out = x / sum_x
        return self.out


    def gradient(self):
        self.eta = self.prediction.copy() - self.label
        return self.eta


# class CrossEntropyLoss():
#     def __init__(self):
#         pass



