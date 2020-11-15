import numpy as np

# 为什么归一化，其实不做归一化也可以训练，但是需要学习率很小（lr=1e-5）。因为学习率一旦稍大，整个网络的权重会变得很大，梯度
# 变得很大，导致梯度爆炸。


# 实现对输入数据的归一化
def normalization(x):
    """
    :param x:输入的数据维度可能是[N,C,H,W]或[N,m]
    :return: 归一化后的结果
    """
    eps = 1e-5
    if x.ndim > 2:
        mean = np.mean(x, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
        var = np.var(x, axis=(0, 2, 3))[:, np.newaxis, np.newaxis]
        x = (x - mean) / np.sqrt(var + eps)
    else:
        mean = np.mean(x, axis=1)[:, np.newaxis]
        var = np.var(x, axis=1)[:, np.newaxis] + eps
        x = (x - mean) / np.sqrt(var)

    return x

if __name__ == "__main__":
    # x = np.random.randint(5, size=(2, 5))
    x = np.random.randint(5, size=(2,2,4,4))
    result = normalization(x)
    print(result)