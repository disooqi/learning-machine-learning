import numpy as np
import scipy.io

class HiddenLayer:
    fff = True
    def __init__(self, n_units, n_in, activation='sigmoid', initializer='1', output_layer=False):

        self.n_units = n_units

        if activation == 'sigmoid':
            self.activation=self.sigmoid
            self.dAdZ = self.sigmoid_prime
        elif activation == 'relu':
            self.activation = self.relu
            self.dAdZ = self.relu_prime
        elif activation == 'tanh':
            self.activation = self.tanh
            self.dAdZ = self.tanh_prime
        elif activation == 'leaky_relu':
            self.activation = self.leaky_relu
            self.dAdZ = self.leaky_relu_prime

        # multiplying W by a small number makes the learning fast
        self.W = np.random.randn(n_units, n_in) * 0.01

        self.b = np.zeros((n_units, 1))

        if output_layer:
            self.output_layer = True
        else:
            self.output_layer = False

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

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
    def __init__(self, X, y):
        self.classes = np.unique(y)
        self.layers = list()

        self.X = X
        # self.y = to_categorical(y)

        incidence_y = np.zeros((self.classes.size, y.size))
        print(incidence_y.shape)
        incidence_y[y.ravel()-1, np.arange(y.size)] = 1  # (5000, 10)

        self.y = incidence_y

    def add_layer(self, n_units, activation='sigmoid', initializer='1', output_layer=False):
        if self.layers:
            layer = HiddenLayer(n_units, self.layers[-1].n_units, activation=activation, initializer=initializer)
        else:
            layer = HiddenLayer(n_units, self.X.shape[0], activation=activation, initializer=initializer)
        self.layers.append(layer)

    def add_output_layer(self, initializer='1'):
        if not self.layers[-1].output_layer:
            self.add_layer(self.classes.size, activation='sigmoid', initializer=initializer, output_layer=True)
        else:
            # TODO: you should raise an error and message that says you need to delete existing output_layer
            pass

    def _calculate_single_layer_gradients(self, dLdA, layer_cache, compute_dJdA_1 = True):
        '''
        :param dJdA:
        :return: dJdA_1, dJdW, dJdb
        '''
        # dz = da * g'(z) TODO: currently we pass A instead of Z, I guess it is much better to follow "A. Ng" and pass Z
        dAdZ = layer_cache.dAdZ(layer_cache.A)
        dLdZ = dLdA * dAdZ

        # dw = dz . a[l-1]
        dZdW = layer_cache.A_l_1
        dJdW = np.dot(dLdZ, dZdW.T) / self.X.shape[1]  # this is two steps in one line; getting dLdw and then dJdW
        dJdb = np.sum(dLdZ, axis=1, keepdims=True) / self.X.shape[1]

        dLdA_1 = None
        if compute_dJdA_1:
            # da[l-1] = w[l].T . dz[l]
            dZdA_1 = layer_cache.W
            dLdA_1 = np.dot(dZdA_1.T, dLdZ) # computing dLd(A-1)
        return dLdA_1, dJdW, dJdb

    def _calculate_gradients_and_update_weights(self, alpha):
        A = self.X
        for layer in self.layers:
            layer.A_l_1 = A   # this is A-1 from last loop step
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
            layer.A = A

        dLdA = -self.y / A + (1 - self.y) / (1 - A)
        # To avoid the confusion: reversed() doesn't modify the list. reversed() doesn't make a copy of the list
        # (otherwise it would require O(N) additional memory). If you need to modify the list use alist.reverse(); if
        # you need a copy of the list in reversed order use alist[::-1]
        for l, layer in zip(range(len(self.layers), 0, -1), reversed(self.layers)):
            # dAdZ = layer.dAdZ(layer.A)
            # dZdW = layer.A_l_1
            #
            # # dLdW = dLdA * dAdZ * dZdW
            # dLdZ = previous * dAdZ
            # dJdW = np.dot(dLdZ, dZdW.T) / self.X.shape[1]  # this is two steps in one line; getting dLdw and then dJdW
            # # dLdW = dLdA * dAdZ
            # dJdb = np.sum(dLdZ, axis=1, keepdims=True) / self.X.shape[1]
            #
            # dZdA_1 = layer.W


            # if l > 1:
            #     dJdA_1 = np.dot(dZdA_1.T, dLdZ) / self.X.shape[1]  # this is two steps in one line; getting dLd(A-1) and then dJd(A-1)
            #     previous = dJdA_1

            dLdA, dJdW, dJdb = self._calculate_single_layer_gradients(dLdA, layer, compute_dJdA_1=(l>1))

            layer.W -= alpha*dJdW
            layer.b -= alpha*dJdb



    def accuracy(self):
        A = self.X
        for layer in self.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
        else:

            y = self.y.argmax(axis=0) + 1
            pred = A.argmax(axis=0) + 1
            res = np.equal(pred, y)
            return 100* np.sum(res)/y.size

    def train(self, alpha=0.01, iterations=1):
        print(self.cost())
        for i in range(iterations):
            self._calculate_gradients_and_update_weights(alpha=alpha)
            if i%10==0:
                print(i, self.cost(), self.accuracy())


        else:
            print(self.cost())

    @staticmethod
    def cross_entropy_loss(y, a):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    def cost(self):
        A = self.X
        for layer in self.layers:
            Z = np.dot(layer.W, A) + layer.b
            A = layer.activation(Z)
        else:
            loss_matrix = self.cross_entropy_loss(self.y, A)
            sum_over_all_examples = np.sum(loss_matrix, axis=1)/loss_matrix.shape[1]
            return np.sum(sum_over_all_examples)/sum_over_all_examples.size
            # print(incidence_y.argmax(axis=0)+1)


if __name__ == '__main__':
    handwritten_digits = scipy.io.loadmat("../data/ex3data1.mat")
    print(handwritten_digits.keys())
    Xf = handwritten_digits['X'].T
    yf = handwritten_digits['y'].T

    print('There are {1:} training examples, each individual example is represented using {0:} features'.format(
        *Xf.shape))
    nn01 = NN(Xf, yf)
    # nn01.add_layer(250, a activation='relu')
    # nn01.add_layer(531, activation='relu')
    # nn01.add_layer(250, activation='leaky_relu')
    # nn01.add_layer(500, activation='tanh')
    # nn01.add_layer(200, activation='tanh')
    # nn01.add_layer(95, activation='tanh')
    nn01.add_layer(50, activation='tanh')

    nn01.add_output_layer()
    nn01.train(iterations=10000, alpha=1)
    print(nn01.accuracy())
