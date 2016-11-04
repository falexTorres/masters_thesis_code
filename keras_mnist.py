import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale
#plt.subplot(331)
#plt.xlabel("label: " + str(y_train[0]))
#plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))

#plt.subplot(332)
#plt.xlabel("label: " + str(y_train[1]))
#plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))

#plt.subplot(333)
#plt.xlabel("label: " + str(y_train[2]))
#plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))

#plt.subplot(338)
#plt.xlabel("label: " + str(y_train[3]))
#plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

#plt.show()

# flatten 28x28 images into a vector with length = 784
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
# print error for baseline model
print "Baseline Error: %.2f%%" % (100 - scores[1] * 100)




