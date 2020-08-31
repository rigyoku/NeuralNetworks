import sgd
import imgProcessing
import numpy as np
import matplotlib.pyplot as mpt

# 取得数据
data = imgProcessing.getData()
input = data.get('input')
testInput = data.get('testInput')
label = data.get('label')
testLabel = data.get('testLabel')

# 对总数据进行1000个抽样
sampling = np.random.choice(input.shape[0], 10)
input = input[sampling]
label = label[sampling]

net = sgd.SgdNet(input.shape[1], label.shape[1], True, 28)
loss = []
correctRate = []
testRate = []
for index in range(0, 3):
    l, c = net.batchLearn(input, label, 5, 0.01, 10)
    loss.extend(l)
    correctRate.append(c)
    testRate.append(net.correctRate(net.output(testInput), testLabel))

x = np.arange(0, len(loss))
mpt.plot(x, loss)
mpt.show()

x = np.arange(0, 3)
mpt.plot(x, correctRate)
mpt.plot(x, testRate, linestyle='--')
mpt.show()
