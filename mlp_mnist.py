'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
import numpy as np
import sys
import pickle

np.set_printoptions(threshold=30)

with open(sys.argv[1], 'rb') as f:
	x_train = pickle.load(f)

with open(sys.argv[2], 'rb') as f:
	x_test = pickle.load(f)

with open(sys.argv[3], 'rb') as f:
	y_train = pickle.load(f)

with open(sys.argv[4], 'rb') as f:
	y_test = pickle.load(f)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

batch_size = 128
num_classes = 10
epochs = 100
input_dim = x_train.shape[1]

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

tensorboard = TensorBoard(log_dir='./logs/'+sys.argv[1][:-4]+'/', histogram_freq=0, write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
