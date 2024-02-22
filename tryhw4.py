import sys
sys.dont_write_bytecode = True
from uwimg import *

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    print(inputs)
    l = [   make_layer(inputs, 64, RELU),
            make_layer(64, 32, RELU),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)

print("loading data...")
train = load_classification_data(b"cifar.train", b"cifar/labels.txt", 1)
test  = load_classification_data(b"cifar.test", b"cifar/labels.txt", 1)
print("done")
print

print("training model...")
batch = 128
iters = 3000
rates = [.01]
momentum = .9
decays = [.01]
bestRateTrain = 0.0
bestDecayTrain = 0.0
bestAccuracyTrain = 0.0
bestRateTest = 0.0
bestDecayTest = 0.0
bestAccuracyTest = 0.0
for rate in rates :
    for decay in decays :
        m = neural_net(train.X.cols, train.y.cols)
        train_model(m, train, batch, iters, rate, momentum, decay)
        if accuracy_model(m, train) > bestAccuracyTrain :
            bestRateTrain = rate
            bestDecayTrain = decay
            bestAccuracyTrain = accuracy_model(m, train)
        
        if accuracy_model(m, test) > bestAccuracyTest :
            bestRateTest = rate
            bestDecayTest = decay
            bestAccuracyTest = accuracy_model(m, test)
            
        print("done")
        print
        print("evaluating model...")
        print("training accuracy: %0.2f %%"%(100*accuracy_model(m, train)))
        print("test accuracy:     %0.2f %%"%(100*accuracy_model(m, test)))
        

print("evaluating best model...")
print("best rate train: ", bestRateTrain)
print("best decay train: ", bestDecayTrain)
print("best rate test: ", bestRateTest)
print("best decay test: ", bestDecayTest)
print("best training accuracy: %0.2f %%"%(100*bestAccuracyTrain))
print("best test accuracy:     %0.2f %%"%(100*bestAccuracyTest))

import IPython
IPython.embed()