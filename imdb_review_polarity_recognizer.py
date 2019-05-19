"""A simple neural network for recognizing the polarity of movie reviews on IMDB's database."""
from keras.datasets import imdb
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def vectorize_sequences(seq, dim=10000):
    """
    A function that creates a one-hot encoding of the input sequence by creating
    a zero-matrix, and setting only the required elements to 1.
    """
    result = np.zeros((len(seq), dim))
    for i, sequence in enumerate(seq):
        result[i, sequence] = 1
    return result


def decode_review(review_num):
    """
    A function that decodes an integer coded movie review.
    """
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[review_num]])
    return decoded_review


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# vectorize the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# vectorize the labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# model definition
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model compilation
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# model validation
x_val = x_train[0:10000]
partial_x_train = x_train[10000:]
y_val = y_train[0:10000]
partial_y_train = y_train[10000:]

# model training
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

if __name__ == '__main__':
    results = model.evaluate(x_test, y_test)
    labels = model.metrics_names
    print(labels)
    print(results)

    # model plotting for loss
    history_dict = history.history
    acc = history_dict['acc']
    loss_vals = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss_vals, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
    plt.title('Training And Validation Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
