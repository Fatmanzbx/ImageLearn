import math
import random
import numpy as np
import sys
import scipy.io as si

class Linear():
    def __init__(self,train_x,train_y,n=10):
        self.train_x = train_x
        self.train_y = train_y
        self.n=n
        m = len(train_y)
        self.weight = np.zeros(m)
        for i in range(m):
            self.weight[i]=1/m
        self.theta=np.zeros((n,train_x.shape[1]))
    '''
    load saved parameter or set the parmeter manually
    '''
    def load(self):
        try:
            self.theta = si.loadmat('theta.mat')['theta']
        except FileNotFoundError as e:
            print('No file find')
            return
    def set_theta(self, theta):
        self.theta=theta
    '''
    Save trained model
    '''
    def save(self):
        si.savemat('theta.mat', {'theta': self.theta})
    """
    To set weight in adaboost
    """
    def set_weight(self,weight):
        self.weight=weight
    """
    loss function
    """
    def LF(self,theta,X,Y):
        m=len(Y)
        sum =0
        for i in range(0,m):
            nor=np.sum(np.exp(np.dot(theta,X[i])))
            for j in range(0,self.n):
                if Y[i] == j:
                    sum+= self.weight[i]*math.log(math.exp(np.dot(theta[j],X[i]))/nor)
                    break
        return -sum
    """
    gradient of each parameter
    """
    def gradLF(self,theta,X,Y,p,q):
        m = len(Y)
        sum = 0
        for i in range(0, m):
            nor=np.sum(np.exp(np.dot(theta,X[i])))
            for j in range(0, self.n):
                if Y[i] == j:
                    if p == j:
                        a=X[i][q] - (math.exp(np.dot(theta[p], X[i])) * X[i][q]) / nor
                    else:
                        a=-(math.exp(np.dot(theta[p],X[i]))*X[i][q]) / nor
                    sum+=a*self.weight[i]
        return -sum
    """
    Use the model to generate Y from X
    """
    def predict(self,x):
        return np.argmax(np.dot(self.theta,x.T))
    """
    Generate the probability of each label
    """
    def prob(self,x):
        nor = np.sum(np.exp(np.dot(self.theta, x)))
        p=np.exp(np.dot(self.theta, x))/nor
        return p

    """
    Load data and count the accuracy
    """
    def error(self):
        k=0
        r=0
        for Im in self.train_x:
            a=self.predict(Im)
            if a!=self.train_y[k]:
                r=r+self.weight[k]
            k=k+1
        return r
    def test(self,Ims,labels):
        k=0
        r=0
        for Im in Ims:
            a=self.predict(Im)
            if a==labels[k]:
                r=r+1
            k=k+1
        return r/k
    def train(self,iter,stride,batch):
        """
        load data
        """
        Ims = self.train_x
        labels = self.train_y
        theta=self.theta
        loss=10
        t=0
        while loss > 0.1 and t < iter:
            """
            Use MBGD
            """
            index = random.sample(range(len(labels)),batch)
            batch_Ims = Ims[index]
            batch_labels = labels[index]
            for i in range(0, self.n):
                print(i)
                for j in range(0,Ims.shape[1]):
                    theta[i][j]-=self.gradLF(theta,batch_Ims,batch_labels,i,j )*stride/(5+t)
                sys.stdout.write('\033[F')
            t+=1
            loss=self.LF(theta,Ims,labels)
            self.theta=theta
            print('iteration {}, loss {}'.format(t, loss))
        return theta