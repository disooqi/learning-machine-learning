import numpy as np
from keras.models import Sequential     # adding elements(i.e. layers) into this model in a sequence
from keras.layers import Dense
from keras.optimizers import Adam, SGD


data = np.loadtxt('../data/ex2data1.txt', delimiter=',')
features = data[:,:2]
y = data[:,2]

print(features.shape)
print(y.shape)


model = Sequential()
model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
model.compile(SGD(lr=0.5), 'binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(features, y, epochs=25)