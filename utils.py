import numpy as np

def sigmoid(x):
    '''
    激活函数sigmoid,进行柔滑
    params:
        x   输入数据
    return:
        激活后数据
    '''
    return 1 / (1 + np.exp(-1 * x))

def relu(x):
    '''
    激活函数ReLU,进行柔滑
    params:
        x   输入数据
    return:
        激活后数据
    '''
    if x > 0:
        return x
    else:
        return 0

def softmax(x):
    '''
    输出函数softmax,进行柔滑
    因为乘方处理数值越大梯度越大,所有进行了缩小处理,每个值均减去最大值.
    由于分子分母同乘固定值不变,设常量c,c*(e^x)=e^logc*e^*=exp(logc+x)=exp(x+C)
    params:
        x   输入数据
    return:
        输出数据或称为概率
    '''
    max = np.max(x)
    if x.ndim == 1:
        sum = np.sum(np.exp(x - max))
    else:
        sum = np.sum(np.exp(x - max), axis=1)
        sum = sum.reshape(sum.size, 1)
    return np.exp(x - max) / sum

def meanSquareError(x, t):
    '''
    均方误差,计算损失
    params:
        x   输出数据
        t   正确数据[one-hot]
    return:
        损失程度
    '''
    return 0.5 * np.sum((x - t) ** 2)

def crossEntropyError(x, t):
    '''
    交叉熵误差,计算损失
    params:
        x   输出数据
        t   正确数据
    return:
        损失程度
    '''
    h = 1e-7
    return -1 * np.sum(t * np.log(x + h))

def partialDerivative(x, f):
    '''
    计算函数f关于变量x的偏导数
    params:
        x   变量
        f   方法
    return:
        函数f关于变量x的偏导数
    '''
    ret = np.zeros_like(x)
    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    h = 1e-4
    while not iterator.finished:
        index = iterator.multi_index
        temp = x[index]
        x[index] = temp + h
        f1 = f(x)
        x[index] = temp - h
        f2 = f(x)
        ret[index] = (f1 - f2) / 2 / h
        iterator.iternext()
    return ret


class SigmoidLayer():

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dOut):
        return dOut * self.out * (1 - self.out)

class Affine():

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        out = np.dot(x, self.W) + self.b
        self.x = x
        return out

    def backward(self, dOut):
        dx = np.dot(dOut, self.W.T)
        self.dW = np.dot(self.x.T, dOut)
        self.db = np.sum(dOut, axis=0)
        return dx

class Swl():

    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        loss = crossEntropyError(self.y, t)
        self.loss = loss
        return loss

    def backward(self):
        size = self.t.shape[0]
        if self.t.ndim == 1:
            size = 1
        return (self.y - self.t) / size
