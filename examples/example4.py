from keras.datasets import boston_housing
from keras import models, layers
from matplotlib import pyplot as plt
import numpy as np

"""
Predicting median prices of homes in a given suburb based on suburb data points
"""


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape) # (404, 13)
print(test_data.shape) # (102, 13)
print("train data", train_data[0])
print("train_targets", train_targets[0])
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)


# subtract the mean of each feature, divide by standard deviation.
# 'best practice' for normalizing data. Centers data around 0
train_data = (train_data - mean)/std


print(mean, std)
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def plot_stuff(plots):
    for plot in plots:
        plt.plot(plot[0], plot[1], 'b', label=plot[3])
        plt.xlabel(plot[4])
        plt.ylabel(plot[5])

# K-fold cross-validation
k = 5
num_val_samples = len(train_data) // k
num_epochs = 70
all_scores = []
all_mae_history = []

for i in range(k):
    print("processing fold #", i)
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],
                                         train_data[(i+1)*num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],
                                            train_targets[(i+1)*num_val_samples:]],
                                           axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1)
    print(history.history.keys())
    print(history.__dict__.keys())
    all_mae_history.append(history.history['val_mae'])
    val_mse, val_mae = model.evaluate(val_data, val_targets)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
print(np.median(all_scores))

average_mae_history = [
    np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)
]

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16)
results = model.evaluate(test_data, test_targets)
print(results)

plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel("Validation MAE")
plt.show()