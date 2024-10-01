# Autoencoder that removes noise/distortion from an image

# Importing packages

import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

# Loading dataset

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data visualization

x_train.shape

# Selecting a data sample

i = random.randint(1, 60000)
plt.imshow(x_train[i], cmap='gray')

label = y_train[i]
print(label)

# Adding noise to the images
# Normalizing

x_train = x_train / 255
y_train = y_train / 255

added_noise = np.random.randn(*(28, 28))
noise_factor = 0.3
added_noise = noise_factor * np.random.randn(*(28, 28))

plt.imshow(added_noise)

# Selecting an image and adding noise

noise_factor = 0.2
sample_image = x_train[41]
noisy_sample_image = sample_image + noise_factor * np.random.randn(*(28, 28))
plt.imshow(noisy_sample_image, cmap='gray')

# Checking min and max values

noisy_sample_image.min()
noisy_sample_image.max()

# Normalizing values between 0 and 1 

noisy_sample_image = np.clip(noisy_sample_image, 0., 1.)

# Verifying

noisy_sample_image.min()
noisy_sample_image.max()

plt.imshow(noisy_sample_image, cmap='gray')

# Doing the same for all images

x_train_noisy = []
noise_factor = 0.2

for sample_image in x_train:
    sample_image_noisy = sample_image + noise_factor * np.random.randn(*(28, 28))
    sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
    x_train_noisy.append(sample_image_noisy)

# Converting the list x_train to a matrix    

x_train_noisy = np.array(x_train_noisy)

plt.imshow(x_train_noisy[3], cmap='gray')

# Doing the same for test images 

x_test_noisy = []
noise_factor = 0.4

for sample_image in x_test:
    sample_image_noisy = sample_image + noise_factor * np.random.randn(*(28, 28))
    sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
    x_test_noisy.append(sample_image_noisy)
    
x_test_noisy = np.array(x_test_noisy)
plt.imshow(x_test_noisy[23], cmap='gray')

# Creating the model

autoencoder = tf.keras.models.Sequential()

# Building the convolutional layer

autoencoder.add(tf.keras.layers.Conv2D(16, (3, 3), strides=1, padding="same", input_shape=(28, 28, 1)))
autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

autoencoder.add(tf.keras.layers.Conv2D(8, (3, 3), strides=1, padding="same"))
autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))

# Encoded image

autoencoder.add(tf.keras.layers.Conv2D(8, (3, 3), strides=1, padding="same"))

# Building the decoding

autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=1, padding="same"))

autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=1, activation='sigmoid', padding="same"))

# Compiling

autoencoder.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

autoencoder.summary()

# Training 

autoencoder.fit(x_train_noisy.reshape(-1, 28, 28, 1),
                x_train.reshape(-1, 28, 28, 1),
                epochs=10,
                batch_size=200)

# Evaluating the model

denoised_images = autoencoder.predict(x_test_noisy[:15].reshape(-1, 28, 28, 1))
denoised_images.shape

fig, axes = plt.subplots(nrows=10, ncols=15, figsize=(30, 6))

for images, row in zip([x_test_noisy[:15], denoised_images], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')
