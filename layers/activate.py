import numpy as np

class Relu():
    def forward(self, x):
        self.x = x
        return np.maximum(self.x, 0)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta


class sigmoid():
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, eta):
        result = eta * (self.out * (1-self.out))
        return result


class tanh():
    def forward(self, x):
        temp1 = np.exp(x) - np.exp(-x)
        temp2 = np.exp(x) + np.exp(-x)
        self.out = temp1 / temp2
        return self.out

    def backward(self, eta):
        return eta * (1 - np.square(self.out))

