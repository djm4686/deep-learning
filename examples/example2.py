from keras.datasets import imdb
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

"""
Classifying imdb movie reviews as positive or negative based on word usage
"""

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
print(train_data[0])
reverse_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_words1 = ' '.join([reverse_index.get(i-3, '?') for i in train_data[0]])
decoded_words2 = ' '.join([reverse_index.get(i-3, '?') for i in train_data[1]])
decoded_words3 = ' '.join([reverse_index.get(i-3, '?') for i in train_data[2]])
print(decoded_words1)
print(decoded_words2)
print(decoded_words3)
print(train_labels[0:3])

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def make_model(parameters, layer_num):
    model = models.Sequential()
    for x in range(layer_num):
        model.add(layers.Dense(parameters, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', "mae"])
    return model

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=1, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

results = model.evaluate(x_test, y_test)

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label="Loss")
plt.plot(epochs, acc, 'b', label="Accuracy")
plt.plot(epochs, val_loss, 'ro', label="Validator Loss")
plt.plot(epochs, val_acc, 'r', label="Validator Acc")
plt.legend()
plt.show()
# epochs = range(1, len(acc) + 1)
#
# print(model.evaluate(x_test, y_test))
#
#
#
# model.predict(x_test)

# def plot_data_list(plots):
#     j = 1
#     for i, plot in enumerate(plots):
#         if i % 12 == 0 and i != 0:
#             j += 1
#         plt.figure(j)
#         plt.subplot(5, 2, i+1)
#         for p, c in zip(plot, ['b', 'bo']):
#             plt.plot(p[0], p[1], c, label=p[2])
#     plt.show()
#
# plots = []
# num_epochs = 1
# i = 0
# for x in [16]:
#     for y in range(1):
#         print("doing model: {} of {}".format(i, 60))
#         model = make_model(x, y)
#         history = model.fit(partial_x_train, partial_y_train, validation_data=(x_val, y_val),
#                             epochs=num_epochs, batch_size=512, verbose=0)
#         history_dict = history.history
#         print(history.history.keys())
#         loss_values = history_dict['loss']
#         val_loss_values = history_dict['val_loss']
#         acc = history_dict['accuracy']
#         val_acc = history_dict['val_accuracy']
#         plots.append([(range(num_epochs), val_loss_values, 'loss values'), (range(num_epochs), val_acc, 'acc values')])
#         i += 1
# plot_data_list(plots)