import gzip
import numpy as np
import pickle
import os

# 资源目录
resource = './resource/'
# 图片大小
img_size = 28 * 28
# 输入数据文件
input = 'train-images-idx3-ubyte.gz'
testInput = 't10k-images-idx3-ubyte.gz'
# 结果数据文件
label = 'train-labels-idx1-ubyte.gz'
testLabel = 't10k-labels-idx1-ubyte.gz'
# pickle存储文件
pic = resource + 'img.pic'

def getInput(filename):
    '''
    读取输入数据文件,由于16进制,offset需要16;通过reshape转成指定列数组;除255做正规化
    params:
        文件名
    return:
        正规化输入数据
    '''
    with gzip.open(resource + filename) as file:
        return np.frombuffer(file.read(), np.uint8, offset=16).reshape(-1, img_size) / 255

def getHots(filename):
    '''
    读取结果文件,1位数字offset为8;转成one-hot形式返回
    params:
        结果文件
    return:
        one-hot
    '''
    with gzip.open(resource + filename) as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)
        hots = np.zeros((data.shape[0], 10))
        for index, val in enumerate(data):
            row = hots[index]
            row[val] = 1
        return hots

def getData():
    '''
    获取训练数据,如果没有缓存文件就直接读取
    return:
        训练数据
    '''
    if os.path.exists(pic):
        with open(pic, 'rb') as p:
            data = pickle.load(p)
    else:
        data = {
            'input': getInput(input),
            'label': getHots(label),
            'testInput': getInput(testInput),
            'testLabel': getHots(testLabel)
        }
        with open(pic, 'wb') as p:
            pickle.dump(data, p, -1)
    return data
