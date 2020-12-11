import random

import scipy.io as si
import numpy as np
import matplotlib.pyplot as plt
import math


def im2col2(input_data, fh, fw, stride=1):
    '''
     Arguments:

     input_data--input,shape:(Number of example,Channel,Height,Width)
     fh -- filter height
     fw --filter width
     stride

     Returns :
     col -- turn sub matrices to vectors to culculate
    '''
    N, C, H, W = input_data.shape

    out_h = (H - fh) // stride + 1
    out_w = (W - fw) // stride + 1

    img = input_data

    col = np.zeros((N, out_h, out_w, fh * fw * C))

    # Turn sub matrices to vectors
    for y in range(out_h):
        y_start = y * stride
        y_end = y_start + fh
        for x in range(out_w):
            x_start = x * stride
            x_end = x_start + fw
            col[:, y, x] = img[:, :, y_start:y_end, x_start:x_end].reshape(N, -1)
    col = col.reshape(N * out_h * out_w, -1)
    return col


def col2im2(col, out_shape, fh, fw, stride=1):
    '''
     Arguments:
     col: vectors
     out_shape-- shape: (Number of example,Channel,Height,Width)
     fh -- filter height
     fw --filter width
     stride

     Returns :
     img -- turn vectors to sub matrices (The reverse of last function)
    '''
    N, C, H, W = out_shape

    col = col.reshape(N, -1, C * fh * fw)
    out_h = (H - fh) // stride + 1
    out_w = (W - fw) // stride + 1
    #print(col.shape)
    #print(out_h)
    img = np.zeros((N, C, H, W))

    # turn col a filter
    for c in range(C):
        for y in range(out_h):
            for x in range(out_w):
                col_index = y * out_w + x
                col_index1 = (c * fh * fw)
                col_index2 = ((c + 1) * fh * fw)
                ih = y * stride
                iw = x * stride
                for k in range(N):
                    img[k, c, ih:ih + fh, iw:iw + fw] = col[k, col_index, col_index1:col_index2].reshape((fh, fw))
    return img


def relu(input_X):
    """
    Arguments:
        input_X -- a numpy array
    Return :
        A: a numpy array. let each elements in array all greater or equal 0
    """

    A = np.where(input_X < 0, 0, input_X)
    return A


def softmax(input_X):
    """
    Arguments:
        input_X -- a numpy array
    Return :
        A: a numpy array same shape with input_X
    """
    exp_a = np.exp(input_X)
    sum_exp_a = np.sum(exp_a, axis=1)
    sum_exp_a = sum_exp_a.reshape(input_X.shape[0], -1)
    ret = exp_a / sum_exp_a
    return ret

def cross_entropy_error(labels, logits):
    return -np.sum(labels * np.log(logits))


class Convolution:
    def __init__(self, W, fb, stride=1):
        """
        Argumets:
        W-- filter weight，shape: (FN,NC,FH,FW),FN is filter number
        fb -- filter bias，shape: (1,FN)
        stride
        """
        self.W = W
        self.fb = fb
        self.stride = stride

        self.col_X = None
        self.X = None
        self.col_W = None

        self.dW = None
        self.db = None
        self.out_shape = None

    #    self.out = None

    def forward(self, input_X):
        """
        input_X-- shape:(m,nc,height,width)
        """
        self.X = input_X
        FN, NC, FH, FW = self.W.shape

        m, input_nc, input_h, input_w = self.X.shape

        out_h = int((input_h - FH) / self.stride + 1)
        out_w = int((input_w - FW) / self.stride + 1)

        self.col_X = col_X = im2col2(self.X, FH, FW, self.stride)

        self.col_W = col_W = self.W.reshape(FN, -1).T
        out = np.dot(col_X, col_W) + self.fb
        out = out.T
        out = out.reshape(m, FN, out_h, out_w)
        self.out_shape = out.shape
        return out

    def backward(self, dz, learning_rate):
        # print("==== Conv backbward ==== ")
        assert (dz.shape == self.out_shape)

        FN, NC, FH, FW = self.W.shape
        o_FN, o_NC, o_FH, o_FW = self.out_shape
        a=self.X.shape
        b=self.col_X.shape
        cz=np.zeros((o_NC,o_FH*o_FW*o_FN))
        for j in range(o_NC):
            cz[j]=dz[:,j,:,:].reshape(1,-1)
        col_dz = cz.T

        self.dW = np.dot(self.col_X.T, col_dz)  # shape is (FH*FW*C,FN)
        s=self.dW.shape
        self.db = np.sum(col_dz, axis=0, keepdims=True)

        self.dW = self.dW.T.reshape(self.W.shape)
        self.db = self.db.reshape(self.fb.shape)

        d_col_x = np.dot(col_dz, self.col_W.T)  # shape is (m*out_h*out_w,FH,FW*C)
        dx = col2im2(d_col_x, self.X.shape, FH, FW, stride=1)
        h=dx.shape
        assert (dx.shape == self.X.shape)

        # refresh and save W, b
        self.W = self.W - learning_rate * self.dW
        self.fb = self.fb - learning_rate * self.db
        #si.savemat('Cov_W.mat', {'W': self.W})
        #si.savemat('Cov_fb.mat', {'fb': self.fb})

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.X = None
        self.arg_max = None

    def forward(self, input_X):
        """
        input_X-- shape: (m,nc,height,width)
        """
        self.X = input_X
        N, C, H, W = input_X.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # transform
        col = im2col2(input_X, self.pool_h, self.pool_w, self.stride)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        arg_max = np.argmax(col, axis=1)
        # maxvalue
        out = np.max(col, axis=1)
        out = out.T.reshape(N, C, out_h, out_w)
        self.arg_max = arg_max
        return out

    def backward(self, dz):
        """
        Arguments:
        dz-- dirivitive of out

        Return:
        dirivitive of input_X
        """
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dz.size, pool_size))

        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dz.flatten()
        dx = col2im2(dmax, out_shape=self.X.shape, fh=self.pool_h, fw=self.pool_w, stride=self.stride)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = X <= 0
        out = X
        out[self.mask] = 0
        return out

    def backward(self, dz):
        dz[self.mask] = 0
        dx = dz
        return dx

class SoftMax:
    def __init__(self):
        self.y_hat = None

    def forward(self, X):
        self.y_hat = softmax(X)
        return self.y_hat

    def backward(self, labels):
        m = labels.shape[0]
        dx = (self.y_hat - labels)

        return dx


def compute_cost(logits, label):
    return cross_entropy_error(label, logits)


class Affine:
    def __init__(self, W, b):
        self.W = W  # shape is (n_x,n_unit)
        self.b = b  # shape is(1,n_unit)
        self.X = None
        self.origin_x_shape = None

        self.dW = None
        self.db = None

        self.out_shape = None

    def forward(self, X):
        self.origin_x_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)  # (m,n)
        out = np.dot(self.X, self.W) + self.b
        self.out_shape = out.shape
        return out

    def backward(self, dz, learning_rate):
        """
        dz-- dirivitive
        """

        assert (dz.shape == self.out_shape)

        m = self.X.shape[0]

        self.dW = np.dot(self.X.T, dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m

        assert (self.dW.shape == self.W.shape)
        assert (self.db.shape == self.b.shape)

        dx = np.dot(dz, self.W.T)
        assert (dx.shape == self.X.shape)

        dx = dx.reshape(self.origin_x_shape)  # 保持与之前的x一样的shape

        # refresh W, b
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
        #si.savemat('Aff_W.mat', {'W': self.W})
        #si.savemat('Aff_b.mat', {'b': self.b})
        return dx


class CNN:

    def __init__(self):
        self.X = None
        self.Y = None
        self.layers = []

    def add_conv_layer(self, n_filter, n_c, f, stride=1):
        """
        add convolution layer
        Arguments:
        n_c -- number of tunnels
        n_filter -- number of filters
        f -- hight,width of filter

        Return :
        Conv -- convolution layer
        """

        # initialize W，fb
        try:
            W = si.loadmat('Cov_W.mat')['W']
            fb = si.loadmat('Cov_fb.mat')['fb']
        except FileNotFoundError as e:
            W = np.random.randn(n_filter, n_c, f, f) * 0.01
            fb = np.zeros((1, n_filter))

        # pooling layer
        Conv = Convolution(W, fb, stride=stride)
        return Conv

    def add_maxpool_layer(self, pool_shape, stride=1):
        """
        add pooling layer
        Arguments:
        pool_shape -- size of filter
        Return :
         Pool -- a pooling layer
        """
        pool_h, pool_w = pool_shape
        pool = Pooling(pool_h, pool_w, stride=stride)

        return pool

    def add_affine(self, n_x, n_units):
        """
        add full connected layer
        Arguments:
        n_x -- number of imput
        n_units -- number of cortexes
        Return :
        fc_layer -- Affine layer
        """

        try:
            W = si.loadmat('Aff_W.mat')['W']
            b = si.loadmat('Aff_b.mat')['b']
        except FileNotFoundError as e:
            W = np.random.randn(n_x, n_units) * 0.01
            b = np.zeros((1, n_units))

        fc_layer = Affine(W, b)

        return fc_layer

    def add_relu(self):
        relu_layer = Relu()
        return relu_layer

    def add_softmax(self):
        softmax_layer = SoftMax()
        return softmax_layer

    def cacl_out_hw(self, HW, f, stride=1):
        return (HW - f) / stride + 1

    def init_model(self, train_X, n_classes):
        """
        initialize a CNN
        """
        N, C, H, W = train_X.shape
        # Convolution
        n_filter = 32
        f = 5

        conv_layer1 = self.add_conv_layer(n_filter=n_filter, n_c=C, f=f, stride=1)

        out_h = self.cacl_out_hw(H, f)
        out_w = self.cacl_out_hw(W, f)
        out_ch = n_filter

        self.layers.append(conv_layer1)

        # Relu
        relu_layer1 = self.add_relu()
        self.layers.append(relu_layer1)

        # Pooling
        f = 2
        pool_layer1 = self.add_maxpool_layer(pool_shape=(f, f), stride=2)
        out_h = self.cacl_out_hw(out_h, f, stride=2)
        out_w = self.cacl_out_hw(out_w, f, stride=2)
        self.layers.append(pool_layer1)

        # Convolution
        n_filter = 64
        f = 5

        conv_layer2 = self.add_conv_layer(n_filter=n_filter, n_c=out_ch, f=f, stride=1)

        out_h = self.cacl_out_hw(out_h, f)
        out_w = self.cacl_out_hw(out_w , f)
        out_ch = n_filter

        self.layers.append(conv_layer2)

        # Relu
        relu_layer2 = self.add_relu()
        self.layers.append(relu_layer2)

        # Pooling
        f = 2
        pool_layer2 = self.add_maxpool_layer(pool_shape=(f, f), stride=2)
        out_h = self.cacl_out_hw(out_h, f, stride=2)
        out_w = self.cacl_out_hw(out_w, f, stride=2)
        self.layers.append(pool_layer2)

        # Affine
        n_x = int(out_h * out_w * out_ch)
        n_units = 512
        fc_layer = self.add_affine(n_x=n_x, n_units=n_units)
        self.layers.append(fc_layer)

        # Relu
        relu_layer = self.add_relu()
        self.layers.append(relu_layer)

        # Affine
        fc_layer = self.add_affine(n_x=n_units, n_units=n_classes)
        self.layers.append(fc_layer)

        # SoftMax
        softmax_layer = self.add_softmax()
        self.layers.append(softmax_layer)

    def forward_progation(self, train_X, print_out=True):
        """
        Arguments:
        train_X: Training Data
        f -- Filter size

        Return :
         Z-- result
         loss -- Loss function value
        """

        N, C, H, W = train_X.shape
        index = 0
        # Convolution
        conv_layer = self.layers[index]
        X = conv_layer.forward(train_X)
        index = index + 1
        if print_out:
            print("After Convolution：" + str(X.shape))

        # Relu
        relu_layer = self.layers[index]
        index = index + 1
        X = relu_layer.forward(X)
        if print_out:
            print("Relu：" + str(X.shape))

        # pooling
        pool_layer = self.layers[index]
        index = index + 1
        X = pool_layer.forward(X)
        if print_out:
            print("Pooling：" + str(X.shape))

        # Convolution
        conv_layer = self.layers[index]
        X = conv_layer.forward(X)
        index = index + 1
        if print_out:
            print("After Convolution：" + str(X.shape))

        # Relu
        relu_layer = self.layers[index]
        index = index + 1
        X = relu_layer.forward(X)
        if print_out:
            print("Relu：" + str(X.shape))

        # pooling
        pool_layer = self.layers[index]
        index = index + 1
        X = pool_layer.forward(X)
        if print_out:
            print("Pooling：" + str(X.shape))

        # Affine
        fc_layer = self.layers[index]
        index = index + 1
        X = fc_layer.forward(X)
        if print_out:
            print("Affline X：" + str(X.shape))

        # Relu
        relu_layer = self.layers[index]
        index = index + 1
        X = relu_layer.forward(X)
        if print_out:
            print("Relu X：" + str(X.shape))

        # Affine
        fc_layer = self.layers[index]
        index = index + 1
        X = fc_layer.forward(X)
        if print_out:
            print("Affline X：" + str(X.shape))

        # SoftMax
        sofmax_layer = self.layers[index]
        index = index + 1
        A = sofmax_layer.forward(X)
        if print_out:
            print("Softmax X：" + str(A.shape))

        return A

    def back_progation(self, train_y, learning_rate):
        """
        Arguments:
        Same as above
        """
        index = len(self.layers) - 1
        sofmax_layer = self.layers[index]
        index -= 1
        dz = sofmax_layer.backward(train_y)

        fc_layer = self.layers[index]
        dz = fc_layer.backward(dz, learning_rate=learning_rate)
        index -= 1

        relu_layer = self.layers[index]
        dz = relu_layer.backward(dz)
        index -= 1

        fc_layer = self.layers[index]
        dz = fc_layer.backward(dz, learning_rate=learning_rate)
        index -= 1

        pool_layer = self.layers[index]
        dz = pool_layer.backward(dz)
        index -= 1

        relu_layer = self.layers[index]
        dz = relu_layer.backward(dz)
        index -= 1

        conv_layer = self.layers[index]
        dz = conv_layer.backward(dz, learning_rate=learning_rate)
        index -= 1

        pool_layer = self.layers[index]
        dz = pool_layer.backward(dz)
        index -= 1

        relu_layer = self.layers[index]
        dz = relu_layer.backward(dz)
        index -= 1

        conv_layer = self.layers[index]
        conv_layer.backward(dz, learning_rate=learning_rate)
        index -= 1

    def optimize(self, train_X, train_y, batch=512, learning_rate=0.05, num_iters=500):
        """
        Arguments:
        train_X -- images
        train_y -- labels
        learning_rate -- learning rate
        num_iters -- max iteration number
        minibatch_size
        """
        m = train_X.shape[0]

        costs = []
        for iteration in range(num_iters):
            iter_cost = 0
            index = random.sample(range(m), batch)
            minibatch_X = train_X[index]
            minibatch_y = train_y[index]

            # foward
            A = self.forward_progation(minibatch_X, print_out=True)
            # loss
            cost = compute_cost(A, minibatch_y)
            # backward
            self.back_progation(minibatch_y, learning_rate)

            if (iteration % 10 == 0):
                print("After %d iters ,cost is :%g" % (iteration, cost/batch))
                costs.append(iter_cost)

        # 画出损失函数图
        plt.plot(costs)
        plt.xlabel("iterations/hundreds")
        plt.ylabel("costs")
        plt.savefig('cost.jpg')

    def predicate(self, train_X):
        """
        Predict
        """
        logits = self.forward_progation(train_X)
        one_hot = np.zeros_like(logits)
        one_hot[range(train_X.shape[0]), np.argmax(logits, axis=1)] = 1
        return one_hot

    def fit(self, train_X, train_y, batch, learn, iter):
        """
        Train
        """
        self.X = train_X
        self.Y = train_y
        n_y = train_y.shape[1]
        m = train_X.shape[0]

        # 初始化模型
        self.init_model(train_X, n_classes=n_y)

        self.optimize(train_X, train_y, batch=batch, learning_rate=learn, num_iters=iter)

        logits = self.predicate(train_X)

        accuracy = np.sum(np.argmax(logits, axis=1) == np.argmax(train_y, axis=1)) / m
        print("训练集的准确率为：%g" % (accuracy))
