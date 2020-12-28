import math

import numpy as np
import scipy.io as si

from BP import BP


class Ada():
    '''
    Inivailize. Start over or load trained data
    '''

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.m = len(train_y)
        self.subtrainers = []
        self.alpha = []
        self.thetas = []
        self.data_weight = np.zeros(self.m)
        for i in range(self.m):
            self.data_weight[i] = 1 / self.m

    def addtrain(self, iter, stride, dim):
        '''
            Add a weak trainer
            Arguments:
                iter: iterations to train weak BP classifier
                stride: learning rate to train weak BP classifier
        '''
        D = self.data_weight
        L = BP()
        L.study_rate = stride
        L.dim_hide = dim
        L.set_weight(D)
        L.init_network(self.train_x / 255.0, self.train_y)
        for t in range(iter):
            L.train(t)
        e = L.error()
        a = 0.5 * math.log((1 - e) / e)
        print(a)
        if a < 0: return a
        self.subtrainers.append((L))
        self.alpha.append(a)
        for j in range(self.m):
            if L.predict(self.train_x[j]) == self.train_y[j]:
                D[j] = D[j] * math.exp(-a)
            else:
                D[j] = D[j] * math.exp(a)
        sumD = np.sum(D)
        for j in range(self.m):
            D[j] = D[j] / sumD
        self.data_weight = D
        return a

    def predict(self, x):
        '''
        Given x, predict y
        '''
        sum = np.zeros(10)
        for i in range(len(self.subtrainers)):
            sum += self.subtrainers[i].forward(x) * self.alpha[i]
        return np.argmax(sum)

    def test(self, Ims, labels):

        '''
        test on test sets
        '''
        k = 0
        r = 0
        for Im in Ims:
            a = self.predict(Im)
            if a == labels[k]:
                r = r + 1
            k = k + 1
        return r / k

    '''
    Save parameter
    '''
    def save(self):
        si.savemat('thetas.mat', {'thetas': self.thetas,'alpha':self.alpha,'data':self.data_weight})

