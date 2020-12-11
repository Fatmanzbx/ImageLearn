import numpy as np
import PrePro as p
from CNN import CNN

def one_hot_label(y):
    one_hot_label = np.zeros((y.shape[0],10))
    y = y.reshape(y.shape[0])
    one_hot_label[range(y.shape[0]),y] = 1
    return one_hot_label

input = 'TrainSet'
Ims, labels = p.load(input,'train')
X_train=Ims/255.0
X_train = X_train.reshape((len(Ims),1,28,28))
y_train = one_hot_label(labels)

input = 'TestSet'
Ims, labels = p.load(input,'t10k')
X_test=Ims/255.0
X_test = X_test.reshape((len(Ims),1,28,28))
y_test = one_hot_label(labels)

convNet = CNN()
train_X = X_train[0:1024]
train_y = y_train[0:1024]
convNet.fit(train_X,train_y,128,0.01,1000)

logits = convNet.predicate(X_test)
m = X_test.shape[0]
accuracy = np.sum(np.argmax(logits,axis=1) == np.argmax(y_test,axis=1))/m
print("测试的准确率为：%g" %(accuracy))