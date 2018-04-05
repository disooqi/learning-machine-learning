import numpy as np
from scipy import optimize


data = np.loadtxt('../data/ex2data1.txt', delimiter=',')

features = data[:,:2]
y = data[:,2]

m, n = features.shape
X = np.column_stack((np.ones((m, 1)), features))
initial_theta = np.zeros((n+1))
# this is the hypothsis function h(X)
def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(initial_theta, X, y):
    # J = (-1/m) * sum(     y .* log(sigmoid(X*theta))     +     (1-y) .* log(1 - sigmoid(X*theta))     );
    h_of_X = sigmoid(np.dot(X, initial_theta[:,np.newaxis]))
    J = np.sum(-y[:, np.newaxis] * np.log(h_of_X) -
                  (1-y[:, np.newaxis]) * np.log(1-h_of_X))/X.shape[0]
    return J

def gradient(theta, X, y):
    # grad = (1/m) *  ( (sigmoid(X*theta) - y)'  * X );
    h_of_X = sigmoid(np.dot(X, theta[:,np.newaxis]))
    grad = np.dot(X.T, h_of_X - y[:, np.newaxis]) / X.shape[0]
    #print grad.flatten()

    return grad.flatten()


print(gradient(initial_theta, X, y))

res = optimize.minimize(fun=costFunction, x0 =initial_theta, args=(X,y), method='TNC', jac=gradient)
print(res)