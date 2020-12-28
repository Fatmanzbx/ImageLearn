from math import sqrt

import numpy as np

import PrePro as pr


class BP:
    def __init__(self):
        print('parameter initializing...')
        self.num_train = 50000
        self.num_confirm = 10000
        self.num_test = 10000
        self.dim_in = 28 * 28
        self.kinds = 10
        self.dim_hide = 30
        self.study_rate = 0.05
        self.loss_limit = 0.1
        self.weight = np.zeros(self.num_train)
        for i in range(self.num_train):
            self.weight[i] = 1 / self.num_train

    def init_network(self, train_x, train_y):
        print('network initializing...')
        self.train_imag_list = train_x[:self.num_train]
        self.train_label_list = train_y[:self.num_train]
        self.confirm_imag_list = train_x[self.num_train:]
        self.confirm_label_list = train_y[self.num_train:]

        self.wjk = (np.random.rand(self.dim_hide, self.kinds) - 0.5) * 2 / sqrt(self.dim_hide)
        self.bj = (np.random.rand(self.kinds) - 0.5) * 2 / sqrt(self.dim_hide)
        self.wij = (np.random.rand(self.dim_in, self.dim_hide) - 0.5) * 2 / sqrt(self.dim_in)
        self.bi = (np.random.rand(self.dim_hide) - 0.5) * 2 / sqrt(self.dim_in)

    def set_weight(self, weight):
        """
        To set weight in adaboost
        """
        self.weight = weight

    def sigmode(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        self.yj = np.dot(x, self.wij) + self.bi
        self.zj = self.sigmode(self.yj)

        self.yk = np.dot(self.zj, self.wjk) + self.bj
        self.zk = self.sigmode(self.yk)
        return self.zk

    def train(self, iter):
        print('Training Round ' + str(iter))
        for circle in range(self.num_train):
            """
            SGD
            """
            sample_i = np.random.randint(0, self.num_train)
            weight = self.weight[sample_i]
            self.forward(self.train_imag_list[sample_i])
            tmp_label = np.zeros(self.kinds)
            tmp_label[int(self.train_label_list[sample_i])] = 1
            print(self.zk)
            delta_k = (self.zk - tmp_label) * self.zk * (1 - self.zk)
            print(tmp_label.shape)
            print(self.zk.shape)
            print(delta_k.shape)
            self.zj.shape = (self.dim_hide, 1)
            delta_k.shape = (1, self.kinds)
            self.wjk = self.wjk - self.study_rate * np.dot(self.zj, delta_k) * weight * self.num_train
            self.zj = self.zj.T
            delta_j = np.dot(delta_k, self.wjk.T) * self.zj * (1 - self.zj)
            print(self.wjk.T.shape)
            print(delta_j.shape)
            print(self.zj.shape)
            print(delta_k.shape)
            tmp_imag = self.train_imag_list[sample_i]
            tmp_imag.shape = (self.dim_in, 1)
            self.wij = self.wij - self.study_rate * np.dot(tmp_imag, delta_j) * weight * self.num_train

    def loss(self):
        """
        Connt Cross Entropy
        """
        ans = 0.0
        for sample_i in range(self.num_confirm):
            self.forward(self.confirm_imag_list[sample_i])
            label_tmp = np.zeros(self.kinds)
            label_tmp[int(self.confirm_label_list[sample_i])] = 1
            ans = ans + sum(np.square(label_tmp - self.zk) / 2.0)
        return ans

    def error(self):
        k = 0
        r = 0
        for Im in self.train_imag_list:
            a = self.predict(Im)
            if a != self.train_label_list[k]:
                r = r + self.weight[k]
            k = k + 1
        return r

    def predict(self, x):
        '''
        Given x, predict y
        '''
        return np.argmax(self.forward(x))

    def test(self, Ims, labels):
        '''
        test on test sets
        '''
        r = 0.0
        for i in range(len(labels)):
            ans = self.predict(Ims[i])
            if ans == labels[i]:
                r = r + 1
        return r / len(labels)


train = 'TrainSet'
test = 'TestSet'
train_x, train_y = pr.load(train, 'train')
test_x, test_y = pr.load(test, 't10k')
data = BP()
data.init_network(train_x / 255.0, train_y)
data.train(0)
