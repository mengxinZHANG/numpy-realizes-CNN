class parameter():
    def __init__(self, w):
        self.data = w     # 权重
        self.grad = None  # 传到下一层的梯度
