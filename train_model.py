import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_train = x_train / 255.0, x_train / 255.0
x_train.shape = (32, 32, 3)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape = (32, 32, 3))
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5)