import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import time

start_time = time.time()

K.set_image_dim_ordering('th')

# fix random seed for reproducibility (causes random function to be predictable)
seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1,28,28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(140, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=400, verbose=2)
# final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

# print results
print "CNN Error: %.2f%%" % (100 - scores[1] * 100)
print model.summary()
print model.get_weights()
model.save("./mnist_cnn_trial")

print "--- %s seconds ---" % (time.time() - start_time)
