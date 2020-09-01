import numpy as np
import utils
import pickle
import os

configPath = './resource/conf.pic'

class SgdNet:

    def __init__(self, inputNum, outputNum, load=False, *hidden):
        '''
        构造方法,初始化权重和偏置
        params:
            self        实例本身
            inputNum    输入层节点数
            outputNum   输出层节点数
            load        是否读取以前的配置
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
            config['W' + str(index + 1)] = np.random.rand(forward, val)
            config['b' + str(index + 1)] = np.zeros(val)
            forward = val
        # 为最后一层权重和参数赋值
        config['W' + str(index + 2)] = np.random.rand(forward, outputNum)
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
        # 计算输出并柔滑
        temp = np.dot(forward, self.config['W' + str(index + 2)]) + self.config['b' + str(index + 2)]
        return utils.softmax(temp)

    def getLoss(self, input, hots):
        '''
        通过交叉熵误差计算损失
        params:
            self        实例本身
            input       输入数据
            hots        正确数据[one-hot形式]
        '''
        out = self.output(input)
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
            每次batch得到的损失函数
        '''
        # 每次batch得到的损失函数
        loss = []
        # 对源数据进行首维度的随机排列
        inDatas = np.random.permutation(inDatas)
        hotDatas = np.random.permutation(hotDatas)
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
        # TODO: append正确率
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


def LayerNet():

    def __init__(self):
        pass

    def derivative(self):
        pass
