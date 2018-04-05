import numpy as np
import scipy.io


class LogisticRegressionClassifier:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.classes = np.unique(y)
        # num_classes = np.size(classes)

        self.n, self.m = X.shape
        self._init_weights_()

    def _init_weights_(self):
        self.classes_weights = dict()
        if np.size(self.classes) > 2:
            for c in self.classes:
                self.classes_weights[c] = {'W': np.zeros((self.n, 1)), 'b': 0}
        elif np.size(self.classes) == 2:
            self.classes_weights['default'] = {'W': np.zeros((self.n, 1)), 'b': 0}

    @staticmethod
    def cross_entropy_loss(y, a):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    def cost(self, W, b):
        Z = np.dot(W.T, self.X) + b
        return np.sum(self.cross_entropy_loss(self.y, self.y_hat(Z))) / self.m

    @staticmethod
    def y_hat(z):
        return 1 / (1 + np.exp(-z))

    def calculate_gradients(self, W, b, X, y):
        pass

    def gradient_descent(self, W, b, X, y):
        dZdW = X  # ----------------------------------------------- (1)
        dZdb = 1  # ----------------------------------------------- (1)

        Z = np.dot(W.T, X) + b

        A = self.y_hat(Z)
        dAdZ = A * (1 - A)  # --------------------------------------(2)

        dLdA = -y / A + (1 - y) / (1 - A)  # ---------------------- (3)

        dLdW = dZdW * dAdZ * dLdA  # ------------------------------ (4)
        dLdb = dZdb * dAdZ * dLdA  # ------------------------------ (4)

        # dW or shortly "np.dot(X,(A-Y).T)/m" and db or shortly "np.sum((A-Y).T)/m"
        dJdW = np.sum(dLdW, axis=1, keepdims=True) / self.m  # --- (5)
        dJdb = np.sum(dLdb, axis=1, keepdims=True) / self.m  # --- (5)

        return dJdW, dJdb

    def gradient_descent_compact(self, W, b, X, y):
        Z = np.dot(W.T, X) + b
        A = self.y_hat(Z)

        dJdW = np.dot(X, (A - y).T) / self.m
        dJdb = np.sum((A-y).T) / self.m

        return dJdW, dJdb

    def _update_weights(self, W, b, X_data, target, alpha=0.0010, iterations=999):
        print("The value of J before learning is", self.cost(W, b), end=', ')
        for i in range(iterations):
            dW, db = self.gradient_descent(W, b, X_data, target)
            W = W - alpha*dW
            b = b - alpha*db
            # print("The value of J after learning is", cost_function(W, b))
        else:
            # print(W, b)
            print("The value of J after learning is", self.cost(W, b))
            return W, b

    def train(self, alpha=0.01, iterations=100000):
        if np.size(self.classes) > 2:
            self.one_vs_all()
        else:
            self.classes_weights['default']['W'], self.classes_weights['default']['b'] = \
                self._update_weights(self.classes_weights['default']['W'], self.classes_weights['default']['b'], self.X, self.y)

        print('Done training!')

    def one_vs_all(self):
        for c in self.classes:
            y_c = np.where(self.y == c, 1, 0)
            self.classes_weights[c]['W'], self.classes_weights[c]['b'] = \
                self._update_weights(self.classes_weights[c]['W'], self.classes_weights[c]['b'], self.X, y_c)




if __name__ == '__main__':
    # handwritten_digits = scipy.io.loadmat("data/ex3data1.mat")
    # print(handwritten_digits.keys())
    # Xf = handwritten_digits['X'].T
    # yf = handwritten_digits['y'].T
    #
    # print('There are {1:} training examples, each individual example is represented using {0:} features'.format(
    #     *Xf.shape))

    file_path = '../data/ex2data1.txt'
    data01 = np.loadtxt(file_path, delimiter=',')

    X2 = data01[:, :-1].T
    y2 = data01[:, -1][np.newaxis, :]
    print('There are {1:} training examples, each individual example is represented using {0:} features'.format(
        *X2.shape))

    # lr = LogisticRegressionClassifier(Xf, yf)
    # lr.train()

    lr2 = LogisticRegressionClassifier(X2, y2)
    lr2.train()

    cost = lr2.cost(np.array([[0.20623159], [0.20147149]]), -25.16131857)
    print(cost)



