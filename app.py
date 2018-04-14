import scipy.io
from classic_NN.nn import NN
from data_preparation.dataset import MNIST_dataset

if __name__ == '__main__':
    handwritten_digits = scipy.io.loadmat("data/ex3data1.mat")
    # print(handwritten_digits.keys())

    mnist = MNIST_dataset(handwritten_digits['X'].T, handwritten_digits['y'].T, dev_size=0.2)

    # print('There are {1:} training examples, each individual example is represented using {0:} features'.format(
    #     *Xf.shape))
    nn01 = NN(n_features=400, n_classes=10, dataset=mnist)
    # nn01.add_layer(531, activation='relu')
    # nn01.add_layer(250, activation='leaky_relu')
    # nn01.add_layer(500, activation='tanh')
    # nn01.add_layer(200, activation='relu')
    # nn01.add_layer(95, activation='relu')
    # nn01.add_layer(2, activation='sigmoid')
    # nn01.add_layer(25, activation='tanh')
    nn01.add_layer(25, activation='relu', dropout_keep_prob=.6)
    nn01.add_layer(25, activation='relu', dropout_keep_prob=.6)
    nn01.add_layer(25, activation='relu', dropout_keep_prob=.6)
    nn01.add_output_layer()

    nn01.train(epochs=1000, mini_batch_size=0, alpha=0.01, regularization_parameter=0)
    print('Dev acc:', nn01.accuracy(mnist.X_dev, mnist.y_dev))
