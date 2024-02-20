
import numpy as np
import tensorflow as tf
from adaptive_densenet import adaptive_densenet
from scipy.misc import imread, imsave, imresize

adn = adaptive_densenet()

image_num = 1

ir_path = '../dataset/dataset_eoir/IR/IR_' + str(image_num) + '.png'
eo_path = '../dataset/dataset_eoir/EO/EO_' + str(image_num) + '.png'

ir_image = imread(ir_path, mode='L')
ir_image = imresize(ir_image, (480, 640), 'nearest', mode='L')
ir_image = ir_image[:,:,np.newaxis]

eo_image = imread(eo_path, mode='RGB')
eo_image = imresize(eo_image, (480, 640), 'nearest', mode='RGB')

input = np.concatenate((eo_image, ir_image),axis=-1)
input = tf.expand_dims(input,axis=0)
input = tf.cast(input, tf.float32)

result_img = adn.dense_net(input)

huber = tf.keras.losses.Huber
