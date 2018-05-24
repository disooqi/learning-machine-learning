import scipy.io
from classic_NN.nn import NN, FullyConnectedLayer
from classic_NN.nn import Optimization
from data_preparation.dataset import MNIST_dataset
import numpy as np

if __name__ == '__main__':

    Z = np.random.randn(6, 1)
    hl = FullyConnectedLayer.softmax(Z)
    iden = np.eye(6)

    sub = hl*(iden-hl)






    handwritten_digits = scipy.io.loadmat("data/ex3data1.mat")
    mnist = MNIST_dataset(handwritten_digits['X'].T, handwritten_digits['y'].T, dev_size=0.2)

    nn01 = NN(n_features=400, n_classes=10)
    nn01.add_layer(25, activation='relu', dropout_keep_prob=1)
    nn01.add_output_layer()

    gd_optimizer = Optimization(method='gradient-descent') # gd-with-momentum gradient-descent rmsprop adam

    gd_optimizer.minimize(nn01, epochs=100, mini_batch_size=512, learning_rate=.1, regularization_parameter=0, dataset=mnist)

    train_acc = nn01.accuracy(mnist.X_train, mnist.y_train)
    dev_acc = nn01.accuracy(mnist.X_dev, mnist.y_dev)
    print('train acc: {:.2f}%, Dev acc: {:.2f}%'.format(train_acc, dev_acc))

