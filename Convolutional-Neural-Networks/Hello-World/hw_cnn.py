from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from PIL import Image 
import numpy as np
import sys

# Loading the MNIST dataset in Keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Defining the network architecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compiling the network
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Preprocessing of the train set and test set
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

# Preparing the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the network
network.fit(train_images, train_labels, epochs=10, batch_size=128)

# Test the accuracy of the network
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test accuracy:',test_acc)
print('Test loss:',test_loss)