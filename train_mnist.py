from mlp import *

import torchvision.datasets as datasets
import struct
from array import array
from os.path import join
import numpy as np  # linear algebra
import random 
import matplotlib
import matplotlib.pyplot as plt

datasets.MNIST(root='./data', train=True, download=True)
datasets.MNIST(root='./data', train=False, download=True)


class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch for labels, expected 2049, got {magic}')
            labels = np.array(array("B", file.read()))  # Convert directly to NumPy array

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch for images, expected 2051, got {magic}')
            image_data = np.frombuffer(file.read(), dtype=np.uint8)  # Read as NumPy array

        # Reshape images to (size, 28, 28)
        images = image_data.reshape(size, rows, cols)

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

input_path = './data/MNIST/raw'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Normalize images to range [0, 1]
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

mlp = MultilayerPerceptron([
    Layer(fan_in=x_train.shape[1], fan_out=256, activation_function=Relu()),
    Layer(fan_in=256, fan_out=128, activation_function=Relu(), dropout_rate=0.03),
    Layer(fan_in=128, fan_out=64, activation_function=Relu(), dropout_rate=0.03),
    Layer(fan_in=64, fan_out=32, activation_function=Relu(), dropout_rate=0.03),
    Layer(fan_in=32, fan_out=10, activation_function=Softmax())  
])


loss_func = CrossEntropy()

training_losses, validation_losses = mlp.train(
    train_x=x_train, train_y=y_train,
    val_x=x_test, val_y=y_test,
    loss_func=loss_func,
    learning_rate=0.0001, batch_size=64, epochs=25,
    rmsprop=True, 
    beta=0.9,      
    epsilon=1e-8   
)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = mlp.forward(x_test, training= False)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test Accuracy: {accuracy:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(training_losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()

# Plot validation loss
plt.subplot(1, 2, 2)
plt.plot(validation_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss Over Time")
plt.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Forward pass to get predictions
y_pred = mlp.forward(x_test, training=False)
y_pred_classes = np.argmax(y_pred, axis=1)  # Predicted class
y_true_classes = np.argmax(y_test, axis=1)  # True labels

# Dictionary to store one sample per class (0-9)
selected_samples = {}

for i in range(len(y_test)):  # Iterate through all test samples
    label = y_true_classes[i]  # Get the true label
    if label not in selected_samples:  # Select the first occurrence of each class
        selected_samples[label] = (x_test[i], y_pred_classes[i])  # Store image and prediction
    if len(selected_samples) == 10:  # Stop once we have one sample per class
        break

# Plot the selected images
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for i, (label, (image, pred)) in enumerate(selected_samples.items()):
    axes[i].imshow(image.reshape(28, 28), cmap="gray")  # Reshape and display
    axes[i].set_title(f"Pred: {pred}")  # Show predicted class
    axes[i].axis("off")

plt.show()
