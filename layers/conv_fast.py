"""
文件conv.py实现的卷积是我们直观理解的卷积，虽然用for循环实现的卷积容易理解，但是速度低下（PS:你要是这么写，
别人肯定会认为你是个low逼）。numpy可以快速实现矩阵运算，所以在实现卷积过程中应尽量避免使用for循环，而是通过
矩阵运算实现。
"""
import numpy as np
from functools import reduce
from layers.parameter import parameter

class conv():
    def __init__(self, filter_shape, stride=1, padding='SAME', bias=True, requires_grad=True):
        """
        :param filter_shape:元组（O, C, K, K）
        :param stride: 步长
        :param padding: 填充方式:{"SAME", "VALID"}
        :param bias:是否有偏置
        :param requires_grad:是否计算梯度
        """
        self.weight = parameter(np.random.randn(*filter_shape) * (2/reduce(lambda x,y:x*y, filter_shape[1:]))**0.5)  #kaiming初始化
        self.stride = stride
        self.padding = padding
        self.requires_grad = requires_grad
        self.output_channel = filter_shape[0]   # 输出通道数
        self.input_channel = filter_shape[1]    # 输入通道数
        self.filter_size = filter_shape[2]  # 卷积核大小
        if bias:
            self.bias = parameter(np.random.randn(self.output_channel))
        else:
            self.bias =None

    def forward(self, input):
        """
        :param input:feature map 形状：[N,C,H,W]
        :return:
        """
        # 第一步边缘填充
        if self.padding == "VALID":
            self.x = input
        if self.padding == "SAME":
            p = self.filter_size // 2
            self.x = np.lib.pad(input, ((0,0),(0,0),(p,p),(p,p)), "constant")
        # 第二步处理输入的宽高不能恰好的被卷积核的大小和选定的步长所整除
        x_fit = (self.x.shape[2] - self.filter_size) % self.stride
        y_fit = (self.x.shape[3] - self.filter_size) % self.stride

        if self.stride > 1:
            if x_fit != 0:
                self.x = self.x[:, :, 0:self.x.shape[2] - x_fit, :]
            if y_fit != 0:
                self.x = self.x[:, :, :, 0:self.x.shape[3] - y_fit]

        # 实现卷积
        N, _, H, W = self.x.shape
        O, C, K, K = self.weight.data.shape
        weight_cols = self.weight.data.reshape(O, -1).T
        x_cols = self.img2col(self.x, self.filter_size, self.filter_size, self.stride)
        result = np.dot(x_cols, weight_cols) + self.bias.data
        output_H, output_W = (H-self.filter_size)//self.stride + 1, (W-self.filter_size)//self.stride + 1
        result = result.reshape((N, result.shape[0]//N, -1)).reshape((N, output_H, output_W, O))
        return result.transpose((0, 3, 1, 2))


    def backward(self, eta, lr):
        """
        :param eta:上一层返回的梯度[N,O,output_H, output_W]
        param lr:学习率
        :return:
        """
        # 在eta行行列列之间插入插入0后，计算W,b的梯度；然后进行padding， padding后计算返回上一层的梯度


        # 第一步步长大于1要在行行和列列之间插0
        if self.stride > 1:
            N, O, output_H, output_W = eta.shape
            inserted_H, inserted_W = output_H + (output_H-1)*(self.stride-1), output_W + (output_W-1)*(self.stride-1)
            inserted_eta = np.zeros((N, O, inserted_H, inserted_W))
            inserted_eta[:,:,::self.stride, ::self.stride] = eta
            eta = inserted_eta

        # 计算本层的W,b的梯度
        N, _, output_H, output_W = eta.shape
        self.b_grad = eta.sum(axis=(0,2,3))
        self.W_grad = np.zeros(self.weight.data.shape)      # 形状[O, C, K, K]
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                self.W_grad[:,:,i,j] = np.tensordot(eta, self.x[:,:,i:i+output_H,j:j+output_W], ([0,2,3], [0,2,3]))
        # 权重更新
        self.weight.data -= lr * self.W_grad / N
        self.bias.data -= lr * self.b_grad / N


        # 第二步边缘填充
        if self.padding == "VALID":
            p = self.filter_size - 1
            pad_eta = np.lib.pad(eta, ((0,0),(0,0),(p,p),(p,p)), "constant", constant_values=0)
            eta = pad_eta
        elif self.padding == "SAME":
            p = self.filter_size // 2
            pad_eta = np.lib.pad(eta, ((0, 0), (0, 0), (p, p), (p, p)), "constant", constant_values=0)
            eta = pad_eta

        # 计算传到上一层的梯度
        _, C, _, _ = self.weight.data.shape
        weight_flip = np.flip(self.weight.data, (2,3))  # 卷积核旋转180度
        weight_flip_swap = np.swapaxes(weight_flip, 0, 1)  # 交换输入、输出通道的顺序[C,O,H,W]
        weight_flip = weight_flip_swap.reshape(C, -1).T
        x_cols = self.img2col(eta, self.filter_size, self.filter_size, self.stride)
        result = np.dot(x_cols, weight_flip)
        N, _, H, W = eta.shape
        output_H, output_W = (H - self.filter_size) // self.stride + 1, (W - self.filter_size) // self.stride + 1
        result = result.reshape((N, result.shape[0] // N, -1)).reshape((N, output_H, output_W, C))
        self.weight.grad = result.transpose((0, 3, 1, 2))

        return self.weight.grad


    def img2col(self, x, filter_size_x, filter_size_y, stride):
        """
        现代计算机运算中矩阵运算已经极为成熟（无论是速度还是内存），因此思路是将x中的每个卷积单位[N,C,K,K]展开为行向量，然后与组成二维矩阵
        与展开的权重进行矩阵乘法。最后把结果reshape一下就可以了。
        缺点：虽然提升了速度，但是增大的内存开销（因为x展开成的二维矩阵，存在大量重复元素）
        :param x:输入的feature map形状：[N,C,H,W]
        :param filter_size_x:卷积核的尺寸x
        :param filter_size_y:卷积核的尺寸y
        :param stride:卷积步长
        :return:二维矩阵 形状：[(H-filter_size+1)/stride * (W-filter_size+1)/stride*N, C * filter_size_x * filter_size_y]
        """

        N, C, H, W = x.shape
        output_H, output_W = (H-filter_size_x)//stride + 1, (W-filter_size_y)//stride + 1
        out_size = output_H * output_W
        x_cols = np.zeros((out_size*N, filter_size_x*filter_size_y*C))
        for i in range(0, H-filter_size_x+1, stride):
            i_start = i * output_W
            for j in range(0, W-filter_size_y+1, stride):
                temp = x[:,:, i:i+filter_size_x, j:j+filter_size_y].reshape(N,-1)
                x_cols[i_start+j::out_size, :] = temp
        return x_cols



