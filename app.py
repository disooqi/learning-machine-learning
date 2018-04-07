import scipy.io
from classic_NN.nn import NN

if __name__ == '__main__':
    handwritten_digits = scipy.io.loadmat("data/ex3data1.mat")
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
    nn01.add_layer(25, activation='leaky_relu')

    nn01.add_output_layer()
    nn01.train(iterations=10000, alpha=0.1)
    print(nn01.accuracy())