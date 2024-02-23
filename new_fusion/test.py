from skimage.transform import resize
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from datetime import datetime
import os
import re


def find_latest_model_file(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith('.keras')]
    # 정규 표현식을 사용하여 날짜와 시간을 추출하는 방법으로 변경
    pattern = re.compile(r'\d{8}-\d{6}')  # YYYYMMDD-HHMMSS 형식 매칭

    def extract_datetime(filename):
        match = pattern.search(filename)
        if match:
            return datetime.strptime(match.group(), "%Y%m%d-%H%M%S")
        return datetime.min  # 매칭되지 않는 경우, 최소 날짜 반환

    latest_file = max(model_files, key=extract_datetime)
    return os.path.join(directory, latest_file)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_nums, batch_size=4, mode='train'):
        self.image_nums = image_nums
        self.batch_size = batch_size
        self.mode = mode
        if mode == 'train':
            self.ir_path = '../dataset/dataset_eoir/IR/'
            self.eo_path = '../dataset/dataset_eoir/EO/'
            self.target_path = '../dataset/seafusion_result/'

        else:
            self.ir_path = '../dataset/testset/IR/'
            self.eo_path = '../dataset/testset/EO/'

    def __len__(self):
        return int(np.ceil(len(self.image_nums) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_nums = self.image_nums[idx * self.batch_size:(idx + 1) * self.batch_size]
        ir_images = []
        eo_images = []
        targets = []

        for image_num in batch_nums:
            ir_image_path = self.ir_path + str(image_num) + '.png'
            eo_image_path = self.eo_path + str(image_num) + '.png'

            ir_image = imageio.imread(ir_image_path, pilmode='L')
            eo_image = imageio.imread(eo_image_path)
            ir_image = resize(ir_image, (480, 640), order=0, preserve_range=True, anti_aliasing=False).astype('float32')
            ir_image = np.expand_dims(ir_image, axis=-1)
            eo_image = resize(eo_image, (480, 640), order=0, preserve_range=True, anti_aliasing=False).astype('float32')
            ir_images.append(ir_image)
            eo_images.append(eo_image)

            if self.mode == 'train':
                target_image_path = self.target_path + str(image_num) + '.png'
                target = imageio.imread(target_image_path)
                target = resize(target, (480, 640), order=0, preserve_range=True, anti_aliasing=False).astype('float32')
                targets.append(target)

        if self.mode == 'train':
            return [np.array(ir_images), np.array(eo_images)], np.array(targets)
        else:
            return [np.array(ir_images), np.array(eo_images)]


