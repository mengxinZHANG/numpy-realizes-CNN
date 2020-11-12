import numpy as np

class Maxpooling():
    def __init__(self, kernel_size=(2, 2), stride=2, ):
        """
        :param kernel_size:池化核的大小(kx,ky)
        :param stride: 步长
        这里有个默认的前提条件就是：kernel_size=stride
        """
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, input):
        """
        :param input:feature map形状[N,C,H,W]
        :return:maxpooling后的结果[N,C,H/ksize,W/ksize]
        """
        N, C, H, W = input.shape
        out = input.reshape(N, C, H//self.stride, self.stride, W//self.stride, self.stride)
        out = out.max(axis=(3,5))
        self.mask = out.repeat(self.ksize[0], axis=2).repeat(self.ksize[1], axis=3) != input
        return out

    def backward(self, eta):
        """
        :param eta:上一层返回的梯度[N,O,H,W]
        :return:
        """
        result = eta.repeat(self.ksize[0], axis=2).repeat(self.ksize[1], axis=3)
        result[self.mask] = 0
        return result

class Averagepooling():
    def __init__(self, kernel_size=(2,2), stride=2):
        self.ksize = kernel_size
        self.stride = stride

    def forward(self, input):
        N, C, H, W = input.shape
        out = input.reshape((N, C, H//self.ksize, self.ksize, W//self.ksize, self.ksize))
        out = out.sum(axis=(3,5))
        out = out / self.ksize**2
        return out

    def backward(self, eta):
        result = eta.repeat(self.ksize, axis=2).repeat(self.ksize, axis=3)
        return result