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

net = sgd.LayerNet(input.shape[1], label.shape[1], False, 0.01, 100)
# net = sgd.LayerNet(input.shape[1], label.shape[1], True)

loss = []
correctRate = []
testRate = []
for index in range(0, 3):
    l, c = net.batchLearn(input, label, 1000, 0.1, 100)
    loss.extend(l)
    correctRate.append(c)
    testRate.append(net.correctRate(net.output(testInput), testLabel))

x = np.arange(0, len(loss))
mpt.plot(x, loss)
mpt.savefig('./loss')
mpt.show()
mpt.cla()

x = np.arange(0, 3)
mpt.plot(x, correctRate)
mpt.plot(x, testRate, linestyle='--')
mpt.savefig('./rate')
mpt.show()
mpt.cla()
