from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Get the train a test sets from the IMDB Dataset
# We'll only keep the top 10000 most frequently occurring
# words in the training data.
(train_data, train_labels), (test_data, test_labels) =imdb.load_data(num_words=10000)

# Encode the training and test sets into binary matrices
# where each row represents which words from the 10000
# are used, 0 indicates the word isn't being used, otherwise
# the value is set to 1.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorize the labels
y_train = np.asarray(train_labels)
y_test = np.asarray(test_labels)

# Let's define the network topology
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc'])

# Create the validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Training the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))

# Plotting the training and validation loss
history_dict =history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(epochs, loss_values, 'bo', label='Training loss')
ax1.plot(epochs, val_loss_values, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Plotting the training and validation accuracy
fig2 = plt.figure()
ax2 = plt.subplot(1, 1, 1)
ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Training acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

plt.show()

results = model.evaluate(x_test, y_test)
print(results)