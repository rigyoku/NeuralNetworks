import numpy as np
import utils
import pickle
import os

configPath = './resource/conf.pic'

"""
class SgdNet:

    def __init__(self, inputNum, outputNum, load=False, weight=0.01, *hidden):
        '''
        构造方法,初始化权重和偏置
        params:
            self        实例本身
            inputNum    输入层节点数
            outputNum   输出层节点数
            load        是否读取以前的配置
            weight      初始化权重的倍率
            *hidden     隐藏层节点数
        '''
        # 参数是True且路径存在则读取pic文件
        if load and os.path.exists(configPath):
            with open(configPath, 'rb') as conf:
                self.config = pickle.load(conf)
                return
        # 权重和偏置的字典
        config = {}
        # 前一层节点数
        forward = inputNum
        # 初始化索引
        index = -1
        # 遍历为隐藏层赋值
        for index,val in enumerate(hidden):
            config['W' + str(index + 1)] = weight * np.random.randn(forward, val)
            config['b' + str(index + 1)] = np.zeros(val)
            forward = val
        # 为最后一层权重和参数赋值
        config['W' + str(index + 2)] = weight * np.random.randn(forward, outputNum)
        config['b' + str(index + 2)] = np.zeros(outputNum)
        # 记录该字典
        self.config = config

    def output(self, input):
        '''
        根据输入数据,计算输出结果
        params:
            self        实例本身
            input       输入数据
        return:
            计算结果
        '''
        # 前一项结果
        forward = input
        # 初始化索引
        index = -1
        # 遍历参数,以w * x + b形式得出结果并进行激活
        for index in range(0, int(len(self.config) / 2) - 1):
            temp = np.dot(forward, self.config['W' + str(index + 1)]) + self.config['b' + str(index + 1)]
            forward = utils.sigmoid(temp)
        # 计算输出
        return np.dot(forward, self.config['W' + str(index + 2)]) + self.config['b' + str(index + 2)]

    def getLoss(self, input, hots):
        '''
        通过交叉熵误差计算损失
        params:
            self        实例本身
            input       输入数据
            hots        正确数据[one-hot形式]
        '''
        out = self.output(input)
        out = utils.softmax(out)
        return utils.crossEntropyError(out, hots)

    def adjust(self, input, hots, learn, times):
        '''
        通过梯度下降法学习来调整参数
        params:
            self        实例本身
            input       输入数据
            hots        正确数据[one-hot形式]
            learn       学习长度
            times       学习次数
        '''
        f = lambda x: self.getLoss(input, hots)
        for index in range(0, times):
            for key in range(0, int(len(self.config) / 2)):
                tempW = self.config['W' + str(key + 1)]
                self.config['W' + str(key + 1)] = tempW - utils.partialDerivative(tempW, f) * learn
                tempB = self.config['b' + str(key + 1)]
                self.config['b' + str(key + 1)] = tempB - utils.partialDerivative(tempB, f) * learn
        with open(configPath, 'wb') as conf:
            pickle.dump(self.config, conf, -1)

    def batchLearn(self, inDatas, hotDatas, length, learn, times):
        '''
        以mini-batch形式进行批量学习,更新一次记录一下损失函数,全部更新完的更新次数为一个epoch,需要评价一次
        params:
            self        实例本身
            inDatas     输入数据集
            hotDatas    正确数据集[one-hot形式]
            length      batch长度
            learn       学习长度
            times       学习次数
        return:
            每次batch得到的损失函数结果和正确率
        '''
        # 每次batch得到的损失函数
        loss = []
        # 对源数据进行首维度的随机排列
        index = np.random.permutation(range(0, inDatas.shape[0]))
        inDatas =inDatas[index]
        hotDatas = hotDatas[index]
        # 从第一位开始取值
        start = 0
        # 开始位置小于数据量进行循环
        while start < inDatas.shape[0]:
            # 结束量为开始位置+batch长度
            end = start + length
            # 超长则到最后一位位置
            if end > inDatas.shape[0]:
                end = inDatas.shape[0]
            # 分别取得当次的batch数据
            index = np.arange(start, end)
            input = inDatas[np.arange(start, end)]
            hots = hotDatas[np.arange(start, end)]
            self.adjust(input, hots, learn, times)
            loss.append(self.getLoss(input, hots))
            # 更新开始位置
            start = end
        correctRate = self.correctRate(self.output(input), hots)
        return loss,correctRate

    def correctRate(self, output, hots):
        '''
        计算正确率
        params:
            self        实例本身
            output      输出数据
            hots        正确数据[one-hot形式]
        return:
            正确率
        '''
        output = np.argmax(output, axis=1)
        hots = np.argmax(hots, axis=1)
        return np.sum(output == hots) / output.shape[0]
"""

class LayerNet:

    def __init__(self, inputNum, outputNum, load=False, weight=0.01, *hidden):
        '''
        构造方法,初始化权重和偏置以及反向传播层级
        params:
            self        实例本身
            inputNum    输入层节点数
            outputNum   输出层节点数
            load        是否读取以前的配置
            weight      初始化权重的倍率
            *hidden     隐藏层节点数
        '''
        # 参数是True且路径存在则读取pic文件
        if load and os.path.exists(configPath):
            with open(configPath, 'rb') as conf:
                self.config = pickle.load(conf)
        # 否则新建权重和偏置
        else:
            # 权重和偏置的字典
            config = {}
            # 前一层节点数
            forward = inputNum
            # 初始化索引
            index = -1
            # 遍历为隐藏层赋值
            for index,val in enumerate(hidden):
                config['W' + str(index + 1)] = weight * np.random.randn(forward, val)
                config['b' + str(index + 1)] = np.zeros(val)
                forward = val
            # 为最后一层权重和参数赋值
            config['W' + str(index + 2)] = weight * np.random.randn(forward, outputNum)
            config['b' + str(index + 2)] = np.zeros(outputNum)
            # 记录该字典
            self.config = config
        # 隐藏层数量
        hiddenSize = int(len(self.config) / 2) - 1
        # 初始化传播层级
        self.initLayer(hiddenSize)

    def initLayer(self, hiddenSize):
        '''
        初始化传播层级,防止更新config时层级信息不变
        params:
            self        实例本身
            inputNum    隐藏层数量
        '''
        # 反向传播层
        layers = []
        # 第一层一定是affine层
        layers.append(utils.Affine(self.config['W1'], self.config['b1']))
        # 遍历隐藏层,每个隐藏层对应一个sigmoid层和affine层
        for index in range(0, hiddenSize):
            layers.append(utils.SigmoidLayer())
            layers.append(utils.Affine(self.config['W' + str(index + 2)], self.config['b' + str(index + 2)]))
        # 记录层级字典
        self.layers = layers

    def output(self, input):
        '''
        根据输入数据,计算输出结果
        对所有Affine和Sigmoid层执行forward
        params:
            self        实例本身
            input       输入数据
        return:
            计算结果
        '''
        temp = input
        for layer in self.layers:
            temp = layer.forward(temp)
        return temp

    def getLoss(self, input, hots):
        '''
        通过交叉熵误差计算损失
        params:
            self        实例本身
            input       输入数据
            hots        正确数据[one-hot形式]
        '''
        out = self.output(input)
        out = utils.softmax(out)
        return utils.crossEntropyError(out, hots)

    def adjust(self, input, hots, learn, times):
        '''
        通过梯度下降法学习来调整参数
        params:
            self        实例本身
            input       输入数据
            hots        正确数据[one-hot形式]
            learn       学习长度
            times       学习次数
        '''
        # 使用softmax-with-loss层求得dl
        swl = utils.Swl()
        for index in range(0, times):
            swl.forward(self.output(input), hots)
            temp = swl.backward()
            # 由于层级是 affine - [sigmoid-affine]*n 形式,倒叙遍历之后偶数项为affine层,奇数项为sigmoid层
            for index, layer in enumerate(reversed(self.layers)):
                if index % 2 == 0:
                    # affine层
                    # 对x的导数用于向前传播
                    temp = layer.backward(temp)
                    confIndex = str(int((len(self.config) - index) / 2))
                    # 权重和偏置的导数直接学习[沿梯度前进]
                    self.config['W' + confIndex] = self.config['W' + confIndex] - layer.dW * learn
                    self.config['b' + confIndex] = self.config['b' + confIndex] - layer.db * learn
                else:
                    # sigmoid层
                    # 只需要向前传播对x的导数
                    temp = layer.backward(temp)
            # 更新完参数需要更新层级信息
            self.initLayer(int((len(self.layers) - 1) / 2))

    def batchLearn(self, inDatas, hotDatas, length, learn, times):
        '''
        以mini-batch形式进行批量学习,更新一次记录一下损失函数,全部更新完的更新次数为一个epoch,需要评价一次
        params:
            self        实例本身
            inDatas     输入数据集
            hotDatas    正确数据集[one-hot形式]
            length      batch长度
            learn       每个batch学习长度
            times       每个batch学习次数
        return:
            损失函数结果和正确率
        '''
        # 损失函数结果
        loss = []
        # 对源数据进行首维度的随机排列
        index = np.random.permutation(range(0, inDatas.shape[0]))
        inDatas =inDatas[index]
        hotDatas = hotDatas[index]
        # 从第一位开始取值
        start = 0
        # 开始位置小于数据量进行循环
        while start < inDatas.shape[0]:
            # 结束量为开始位置+batch长度
            end = start + length
            # 超长则到最后一位位置
            if end > inDatas.shape[0]:
                end = inDatas.shape[0]
            # 分别取得当次的batch数据
            index = np.arange(start, end)
            input = inDatas[np.arange(start, end)]
            hots = hotDatas[np.arange(start, end)]
            self.adjust(input, hots, learn, times)
            loss.append(self.getLoss(input, hots))
            # 更新开始位置
            start = end
        # 计算当前数据正确率
        correctRate = self.correctRate(self.output(input), hots)
        # 保存学习结果
        with open(configPath, 'wb') as conf:
            pickle.dump(self.config, conf, -1)
        return loss,correctRate

    def correctRate(self, output, hots):
        '''
        计算正确率
        params:
            self        实例本身
            output      输出数据
            hots        正确数据[one-hot形式]
        return:
            正确率
        '''
        output = np.argmax(output, axis=1)
        hots = np.argmax(hots, axis=1)
        return np.sum(output == hots) / output.shape[0]
