from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

"""
Classifying reuters articles into 46 different categories based on word usage
"""

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data), len(test_data))
print(train_data[0])
print(train_labels[0])

word_index = reuters.get_word_index()
reverse_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_words = ' '.join([reverse_index.get(i-3, '?') for i in train_data[0]])
decoded_words2 = ' '.join([reverse_index.get(i-3, '?') for i in train_data[1]])
decoded_words3 = ' '.join([reverse_index.get(i-3, '?') for i in train_data[2]])
print(decoded_words)
print(decoded_words2)
print(decoded_words3)
print(train_labels[0:3])

def vectorize_sequence(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)


#same as to_categorical func
def to_one_hot(labels, dimensions=46):
    results = np.zeros((len(labels), dimensions))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax')) #probability distribution on all 46 adding to 1

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=512, epochs=9,
                    validation_data=(x_val, y_val))
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)

results = model.evaluate(x_test, one_hot_test_labels)
print(results)


plt.plot(epochs, loss, 'bo', label="Loss")
plt.plot(epochs, acc, 'b', label="Accuracy")
plt.plot(epochs, val_loss, 'ro', label="Validator Loss")
plt.plot(epochs, val_acc, 'r', label="Validator Acc")

plt.legend()

plt.show()