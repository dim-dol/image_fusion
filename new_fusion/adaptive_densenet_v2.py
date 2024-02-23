import tensorflow as tf
import numpy as np
from tensorflow import keras

concatenate_layer = keras.layers.Concatenate(axis=-1)

ir_input = keras.layers.Input(shape=(480, 640, 1), name='ir_input')
eo_input = keras.layers.Input(shape=(480, 640, 3), name='eo_input')
merged = concatenate_layer([ir_input, eo_input])


conv_0 = keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding="SAME")(merged)
conv_0 = keras.layers.LeakyReLU()(conv_0)

dense_1 = keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding="SAME")(conv_0)
dense_1 = concatenate_layer([dense_1, conv_0])
dense_1 = keras.layers.LeakyReLU()(dense_1)

dense_2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="SAME")(dense_1)
dense_2 = concatenate_layer([dense_2, dense_1])
dense_2 = keras.layers.LeakyReLU()(dense_2)

dense_3 = keras.layers.Conv2D(filters=48, kernel_size=3, strides=(1,1), padding="SAME")(dense_2)
dense_3 = concatenate_layer([dense_3, dense_2])
dense_3 = keras.layers.LeakyReLU()(dense_3)

dense_4 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding="SAME")(dense_3)
dense_4 = concatenate_layer([dense_4, dense_3])
dense_4 = keras.layers.LeakyReLU()(dense_4)

dense_5 = keras.layers.Conv2D(filters=80, kernel_size=3, strides=(1,1), padding="SAME")(dense_4)
dense_5 = concatenate_layer([dense_5, dense_4])
dense_5 = keras.layers.LeakyReLU()(dense_5)

conv_1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="SAME")(dense_5)
conv_1 = keras.layers.LeakyReLU()(conv_1)

conv = keras.layers.Conv2D(filters=3, kernel_size=3, strides=(1,1), padding="SAME", activation="tanh")(conv_1)

model = keras.Model(inputs=[ir_input, eo_input], outputs=conv, name="dense_fusion")

model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['accuracy'])

model.summary()