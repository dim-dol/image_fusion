import keras.src.losses
import numpy as np
from adaptive_densenet_v2 import model
import imageio.v2 as imageio
from datetime import datetime
from test import DataGenerator, find_latest_model_file
import os

IS_TRAIN = 0

train_image_set = 1500
test_image_set = 10

cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    if IS_TRAIN:
        # 모델 훈련
        image_nums = list(range(0, train_image_set))
        data_generator = DataGenerator(image_nums, batch_size=5, mode='train')
        history = model.fit(data_generator, epochs=5)
        history.history

        # 모델 테스트
        test_data_generator = DataGenerator(image_nums, batch_size=1, mode='train')

        # 모델 평가
        loss, accuracy = model.evaluate(test_data_generator)
        print(f"Loss: {loss}, Accuracy: {accuracy}")

        model.save(f'./outputs/model_{cur_time}.keras')

    else:
        # 모델 예측/융합이미지 생성
        os.mkdir('./outputs/images/' + cur_time)

        latest_model_path = find_latest_model_file('./outputs')
        load_model = keras.models.load_model(latest_model_path)

        image_nums = list(range(0, test_image_set))
        test_data_generator = DataGenerator(image_nums, batch_size=5, mode='test')

        img_count = 0
        for batch_index in range(len(test_data_generator)):
            ir_images, eo_images = test_data_generator[batch_index]  # 각 배치 데이터 로드
            predictions = load_model.predict([ir_images, eo_images])  # 모델 예측

            for prediction in predictions:
                filepath = f'./outputs/images/{cur_time}/prediction_{img_count}.png'
                imageio.imwrite(filepath, np.clip(prediction * 255.0, 0, 255).astype(np.uint8))
                img_count += 1

        print("모든 예측 이미지가 저장되었습니다.")


if __name__ == '__main__':
    main()