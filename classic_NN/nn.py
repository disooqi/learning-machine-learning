import numpy as np
# from scipy.special import expit, logit
import time


class HiddenLayer:
    def __init__(self, n_units, n_in, activation='sigmoid', output_layer=False, keep_prob=1):

        self.n_units = n_units
        #  It means at every iteration you shut down each neurons of the layer with "1-keep_prob" probability.
        self.keep_prob = keep_prob

        if activation == 'sigmoid':
            self.activation=self.sigmoid
            self.dAdZ = self.sigmoid_prime
            self._weights_initialization(n_in)
        elif activation == 'relu':
            self.activation = self.relu
            self.dAdZ = self.relu_prime
            self._He_initialization(n_in)
        elif activation == 'tanh':
            self.activation = self.tanh
            self.dAdZ = self.tanh_prime
            self._Xavier_initialization(n_in)
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
            self.dAdZ = self.leaky_relu_prime
            self._He_initialization(n_in)

        self.output_layer = output_layer

    def _weights_initialization(self, n_in):
        # multiplying W by a small number makes the learning fast
        # however from a practical point of view when multiplied by 0.01 using l>2 the NN does not converge
        # that is beacuse it runs into gradients vanishing problem
        self.W = np.random.randn(self.n_units, n_in) * 0.01
        self.b = np.zeros((self.n_units, 1))

    def _He_initialization(self, n_in):
        self.W = np.random.randn(self.n_units, n_in) * np.sqrt(2/n_in)
        self.b = np.zeros((self.n_units, 1))

    def _Xavier_initialization(self, n_in):
        """Initialize weight W using Xavier Initialization

        So if the input features of activations are roughly mean 0 and standard variance and variance 1 then this would
        cause z to also take on a similar scale and this doesn't solve, but it definitely helps reduce the vanishing,
        exploding gradients problem because it's trying to set each of the weight matrices W so that it's not
        too much bigger than 1 and not too much less than 1 so it doesn't explode or vanish too quickly.
        """
        self.W = np.random.randn(self.n_units, n_in) * np.sqrt(1 / n_in)
        self.b = np.zeros((self.n_units, 1))

    def _Benjio_initialization(self, n_in):
        self.W = np.random.randn(self.n_units, n_in) * np.sqrt(2/(n_in+self.n_units))
        self.b = np.zeros((self.n_units, 1))

    @staticmethod
    def sigmoid(Z):
        # https://docs.scipy.org/doc/scipy/reference/generated /scipy.special.expit.html
        return 1 / (1 + np.exp(-Z))
        # return expit(np.clip(Z, -709, 36.73))

    @classmethod
    def sigmoid_prime(cls, A):
        '''
        dAdZ
        '''
        return A * (1 - A)

    @staticmethod
    def tanh(Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    @classmethod
    def tanh_prime(cls, A):
        return 1 - A**2

    @staticmethod
    def relu(Z):
        # a[a<0] = 0
        # return np.clip(Z, 0, Z)
        return np.maximum(Z, 0)

    @staticmethod
    def relu_prime(A):
        A[A > 0] = 1
        return A

    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        '''
        :param Z:
        :param alpha: Slope of the activation function at x < 0.
        :return:

        '''
        #return np.clip(Z, alpha * Z, Z)
        return np.where(Z < 0, alpha*Z, Z)

    @staticmethod
    def leaky_relu_prime(A, alpha=0.01):
        return np.where(A > 0, 1, alpha)

class NN:
    def __init__(self, X, y, loss='cross_entropy'):
        if loss == 'cross_entropy':
            self.loss = self.cross_entropy_loss
            self.activation_prime = self.cross_entropy_prime

        self.classes = np.unique(y)
        self.layers = list()

        self.X = X
        self.m = X.shape[1]
        # self.y = to_categorical(y)

        incidence_y = np.zeros((self.classes.size, y.size))
        incidence_y[y.ravel()-1, np.arange(y.size)] = 1  # (5000, 10)

        self.y = incidence_y

    def add_layer(self, n_units, activation='sigmoid', dropout_keep_prob=1):
        if self.layers:
            n_units_previous_layer = self.layers[-1].n_units
        else:
            n_units_previous_layer = self.X.shape[0]

        layer = HiddenLayer(n_units, n_units_previous_layer, activation=activation, keep_prob=dropout_keep_prob)

        self.layers.append(layer)

    def add_output_layer(self):
        if not self.layers[-1].output_layer:
            self.add_layer(self.classes.size, activation='sigmoid')
            self.layers[-1].output_layer = True
        else:
            # TODO: you should raise an error and message that says you need to delete existing output_layer
            pass

    def _calculate_single_layer_gradients(self, dLdA, layer_cache, compute_dJdA_1=True):
        '''
        :param dJdA:
        :return: dJdA_1, dJdW, dJdb
        '''
        # For the first iteration where loss is cross entropy and activation func of output layer
        # is sigmoid, that could be shorten to,
        # dZ[L] = A[L]-Y
        # In general, you can compute dZ as follows
        # dZ = dA * g'(Z) TODO: currently we pass A instead of Z, I guess it is much better to follow "A. Ng" and pass Z

        # During forward propagation, you had divided A1 by keep_prob. In backpropagation, you'll therefore have to
        # divide dA1 by keep_prob again (the calculus interpretation is that if  A[1]A[1]  is scaled by keep_prob, then
        # its derivative  dA[1]dA[1]  is also scaled by the same keep_prob).
        dLdA = np.multiply(dLdA, layer_cache.D)/layer_cache.keep_prob
        dAdZ = layer_cache.dAdZ(layer_cache.A)
        dLdZ = dLdA * dAdZ  # Element-wise product

        # dw = dz . a[l-1]
        dZdW = layer_cache.A_l_1
        dJdW = np.dot(dLdZ, dZdW.T) / self.m  # this is two steps in one line; getting dLdw and then dJdW
        dJdb = np.sum(dLdZ, axis=1, keepdims=True) / self.m

        dLdA_1 = None
        if compute_dJdA_1:
            # da[l-1] = w[l].T . dz[l]
            dZdA_1 = layer_cache.W
            dLdA_1 = np.dot(dZdA_1.T, dLdZ)  # computing dLd(A-1)
        return dLdA_1, dJdW, dJdb

    def _calculate_gradients_and_update_weights(self, alpha, lmbda):
        A = self.X
        for layer in self.layers:
            layer.A_l_1 = A   # this is A-1 from last loop step
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)

            # NB! we don't not apply dropout to the input layer or output layer.
            D = np.random.rand(*A.shape) <= layer.keep_prob  # dropout
            A = np.multiply(A, D) / layer.keep_prob  # inverted dropout

            layer.D = D
            layer.A = A


        with np.errstate(invalid='raise'):
            try:
                dLdA = self.activation_prime(self.y, A)
            except FloatingPointError:
                # print(Z[A==1], HiddenLayer.sigmoid(Z[A==1]))
                # print(np.sum(np.equal(A, np.ones_like(A))))
                raise
        # To avoid the confusion: reversed() doesn't modify the list. reversed() doesn't make a copy of the list
        # (otherwise it would require O(N) additional memory). If you need to modify the list use alist.reverse(); if
        # you need a copy of the list in reversed order use alist[::-1]
        for l, layer in zip(range(len(self.layers), 0, -1), reversed(self.layers)):
            dLdA, dJdW, dJdb = self._calculate_single_layer_gradients(dLdA, layer, compute_dJdA_1=(l>1))

            # L2 Regularization
            weight_decay = 1 - alpha*lmbda/self.m

            layer.W = weight_decay*layer.W - alpha*dJdW
            layer.b -= alpha*dJdb

    def train(self, alpha=0.01, iterations=1, regularization_parameter=0):
        bef = time.time()
        for i in range(iterations):
            if i%100==0:
                print('Iter # {} error: {:.5f}, training acc: {:.2f}%'.format(i,
                                                  self.cost(regularization_parameter=regularization_parameter),
                                                  self.accuracy(self.X, self.y.argmax(axis=0) + 1)))

            self._calculate_gradients_and_update_weights(alpha=alpha, lmbda=regularization_parameter)
        else:
            aft = time.time()

            print('-'*80)
            print('| Summary')
            print('-'*80)
            print('training time: {:.2f} SECs'.format(aft-bef))
            print('-'*80)
            print('Finish error: {:.5f}, training acc: {:.2f}%'.format(
                self.cost(regularization_parameter=regularization_parameter),
                self.accuracy(self.X, self.y.argmax(axis=0) + 1)))

    @staticmethod
    def cross_entropy_loss(y, a):
        # http://christopher5106.github.io/deep/learning/2016/09/16/about-loss-functions-multinomial-logistic-logarithm-cross-entropy-square-errors-euclidian-absolute-frobenius-hinge.html
        # https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def cross_entropy_prime(y, a):
        return -y/a + (1-y)/(1-a)

    def regularization_term(self, lmbda):
        agg = 0
        for layer in self.layers:
            agg = np.sum(np.square(layer.W))
        else:
            return lmbda/(2*self.m) * agg

    def cost(self, regularization_parameter=0):
        A = self.X
        for layer in self.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
        else:
            loss_matrix = self.loss(self.y, A)
            sum_over_all_examples = np.sum(loss_matrix, axis=1)/loss_matrix.shape[1]
            return np.sum(sum_over_all_examples)/sum_over_all_examples.size + self.regularization_term(lmbda=regularization_parameter)

    def accuracy(self, X, y):
        # You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
        A = X
        for layer in self.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
        else:
            # y = y.argmax(axis=0) + 1
            prediction = A.argmax(axis=0) + 1
            res = np.equal(prediction, y)
            return 100 * np.sum(res)/y.size


if __name__ == '__main__':
    pass
