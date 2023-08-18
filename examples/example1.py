
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from matplotlib import pyplot as plt

"""
Classifying mnist 'handwritten' images as numbers
"""

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]
print(train_labels[4])
print(train_images[4])
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

network = models.Sequential()
network.add(layers.Dense(785, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(len(train_images))
x_val = train_images[:30000]
partial_x_train = train_images[30000:]

y_val = train_labels[:30000]
partial_y_train = train_labels[30000:]

history = network.fit(partial_x_train, partial_y_train, epochs=10, batch_size=128, validation_data=[x_val, y_val])



test_loss, test_acc = network.evaluate(test_images, test_labels)

print("Loss: {}, Accuracy: {}".format(test_loss, test_acc))
print(history.history.keys())
history_dict = history.history
loss = history_dict['loss']
print(loss)
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label="Loss")
plt.plot(epochs, acc, 'b', label="Accuracy")
plt.plot(epochs, val_loss, 'ro', label="Validator Loss")
plt.plot(epochs, val_acc, 'r', label="Validator Acc")
plt.legend()
plt.show()