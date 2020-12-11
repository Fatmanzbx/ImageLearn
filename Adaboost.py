from Linear import Linear
import numpy as np
import math
import scipy.io as si

class Ada():
    '''
    Inivailize. Start over or load trained data
    '''
    def __init__(self,train_x,train_y,load=True):
        self.train_x = train_x
        self.train_y = train_y
        self.m = len(train_y)
        self.subtrainers = []
        if load:
            try:
                self.thetas = si.loadmat('thetas.mat')['thetas']
                self.alpha = si.loadmat('thetas.mat')['alpha']
                self.data_weight = si.loadmat('thetas.mat')['data']
            except FileNotFoundError as e:
                print('No saved file foundd')
                return
            for theta in self.thetas:
                L = Linear(self.train_x, self.train_y)
                L.set_theta(theta)
                self.subtrainers.append(L)
        else:
            self.alpha=[]
            self.thetas = []
            self.data_weight = np.zeros(self.m)
            for i in range(self.m):
                self.data_weight[i]=1/self.m

    '''
    Add a weak trainer
    '''
    def train(self,iter, stride, batch):
        D=self.data_weight
        L=Linear(self.train_x, self.train_y)
        L.set_weight(D)
        self.thetas.append(L.train(iter,stride,batch))
        self.subtrainers.append((L))
        e=L.error()
        a=0.5*math.log((1-e)/e)
        print(a)
        self.alpha.append(a)
        for j in range(self.m):
            if L.predict(self.train_x[j])==self.train_y[j]:
                D[j] = D[j] * math.exp(-a)
            else:
                D[j] = D[j] * math.exp(a)
        sumD=np.sum(D)
        for j in range(self.m):
            D[j]=D[j]/sumD
        self.data_weight=D

    '''
    Given x, predict y
    '''
    def predict(self,x):
        sum=np.zeros(10)
        for i in range(len(self.subtrainers)):
            sum+=self.subtrainers[i].prob(x)*self.alpha[i]
        return np.argmax(sum)

    '''
    test on test sets
    '''
    def test(self,Ims,labels):
        k=0
        r=0
        for Im in Ims:
            a=self.predict(Im)
            if a==labels[k]:
                r=r+1
            k=k+1
        return r/k

    '''
    Save parameter
    '''
    def save(self):
        si.savemat('thetas.mat', {'thetas': self.thetas,'alpha':self.alpha,'data':self.data_weight})

