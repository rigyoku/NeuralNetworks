import sgd
import imgProcessing
import numpy as np
import matplotlib.pyplot as mpt

data = imgProcessing.getData()
input = data.get('input')
testInput = data.get('testInput')
label = data.get('label')
testLabel = data.get('testLabel')

net = sgd.SgdNet(input.shape[1], label.shape[1], 5)
loss = []
correctRate = []
testRate = []
for index in range(0, 3):
    l, c = net.batchLearn(input, label, 10000, 0.01, 10)
    loss.append(l)
    correctRate.append(c)
    testRate.append(net.correctRate(net.output(testInput), testLabel))

x = np.arange(0, len(loss))
mpt.plot(x, loss)
mpt.show()

x = np.arange(0, 3)
mpt.plot(x, correctRate)
mpt.plot(x, testRate, linestyle='--')
mpt.show()
