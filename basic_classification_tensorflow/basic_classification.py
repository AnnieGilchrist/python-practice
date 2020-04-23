import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# There are 60,000 images in the training set, each image represented as 28x28 pixels --> print(train_images.shape)

# There are 60,000 labels in the training set --> print(len(train_labels))

# Each label is an integer between 0 and 9, representing a different type of clothing (see class names)

# There are 10,000 images in the test set, each 28 x 28 pixels --> print(test_images.shape)

# And the test set contains 10,000 image labels --> print(len(test_labels))

# This shows us a heatmap of the first image in the training set. Pixel values fall in the range of 0 to 255.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
plt.clf()

# We need to scale these values to a range of 0 to 1...
train_images = train_images / 255.0
test_images = test_images / 255.0

# Print to display teh first 25 images...
plt.figure(figsize=(10,10))

for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])

plt.show()

plt.clf()

# Building the model, we need to set up the layers of our neural network...

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# The first layer, 'keras.layers.Flatten(input_shape=(28, 28))', transforms the images from a two-dimensional array of 28 by 28 pixels to a one-dimmensional array of 28 * 28 = 784 pixels. Then we have two Dense layers. These are densely connected (or fully connected) neural layers. The first Dense layer has 128 nodes (or neurons), the second returns a logits array with length 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.

# Now we need to compile the model. It needs:
#     - Loss function: measures how accurate the model is during training, we want to minimise this function to steer the model in the right direction.
#     - Optimizer: how the model is updated basesd on the data it sees and its loss function.
#     - Metrics: used to monitor training and testing steps. This example uses accuracy, e.g. fraction of images correctly classified.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Now train and test the model...

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# This model produces linear outputs called logits. Attaching a softmax layer to the model allows us to convert these to probabilities which are easier to interpret.

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# If we now print the first prediction for the test_images, we can see an array of 10 numbers which show the confidence the model has that the image corresponds to each of the 10 different types of clothing.

print(predictions[0])

# This will give us the clothing label which has the highest confidence...

print(class_names[np.argmax(predictions[0])])


# Taking a better look at what's going on:

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

