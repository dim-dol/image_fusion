import keras.src.losses
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import tensorflow as tf
from adaptive_densenet_v2 import model
import imageio.v2 as imageio

image_set = 10

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_nums, batch_size=4):
        self.image_nums = image_nums
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_nums) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_nums = self.image_nums[idx * self.batch_size:(idx + 1) * self.batch_size]
        ir_images = []
        eo_images = []
        targets = []

        for image_num in batch_nums:
            ir_path = '../dataset/dataset_eoir/IR/' + str(image_num) + '.png'
            eo_path = '../dataset/dataset_eoir/EO/' + str(image_num) + '.png'
            target_path = '../dataset/seafusion_result/' + str(image_num) + '.png'

            ir_image = imageio.imread(ir_path, pilmode='L')
            eo_image = imageio.imread(eo_path)
            target = imageio.imread(target_path)

            ir_image = resize(ir_image, (480, 640), order=0, preserve_range=True, anti_aliasing=False).astype('float32')
            ir_image = np.expand_dims(ir_image, axis=-1)
            eo_image = resize(eo_image, (480, 640), order=0, preserve_range=True, anti_aliasing=False).astype('float32')
            target = resize(target, (480, 640), order=0, preserve_range=True, anti_aliasing=False).astype('float32')

            ir_images.append(ir_image)
            eo_images.append(eo_image)
            targets.append(target)

        return [np.array(ir_images), np.array(eo_images)], np.array(targets)




# 모델 훈련
image_nums = list(range(0, 20))


data_generator = DataGenerator(image_nums, batch_size=5)
history = model.fit(data_generator, epochs=10)

history.history

results = model.evaluate()