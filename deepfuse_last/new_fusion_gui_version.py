import cv2

import matplotlib.pyplot as plt
import os


loop = 1
th = 130
image_num = 1
image_save = 1   # each image save : 1

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def image_process():
    global th, image_num
    fus_y = cv2.imread('./outputs/fused' + str(image_num) + '_deepfuse_addition1.png', cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread('../dataset/dataset_eoir/IR/IR_' + str(image_num) + '.png', cv2.IMREAD_GRAYSCALE)
    eo = cv2.imread('../dataset/dataset_eoir/EO/EO_' + str(image_num) + '.png', cv2.IMREAD_COLOR)

    color_ycbcr = cv2.cvtColor(eo, cv2.COLOR_BGR2YCrCb)

    y, cb, cr = cv2.split(color_ycbcr[:, :, 0:3])

    # Apply thresholding to the infrared image
    infrared_mask = cv2.threshold(ir, th, 255, cv2.THRESH_BINARY)[1]

    # Apply the mask to the RGB image
    masked_rgb_img = cv2.bitwise_and(ir, fus_y, mask=infrared_mask)
    # masked_rgb_img=masked_rgb_img*1.2
    new_gray = masked_rgb_img

    for i in range(fus_y.shape[0]):
        for j in range(fus_y.shape[1]):
            if masked_rgb_img[i, j] == 0:
                new_gray[i, j] = y[i, j]

    fusion = cv2.merge((fus_y, cb, cr))
    fusion = cv2.cvtColor(fusion, cv2.COLOR_YCrCb2BGR)

    new_fusion = cv2.merge((masked_rgb_img, cb, cr))
    new_fusion = cv2.cvtColor(new_fusion, cv2.COLOR_YCrCb2BGR)

    return eo, fus_y, y, ir, fusion, new_gray, new_fusion, masked_rgb_img

def print_img(axs):
    global eo, fus_y, y, ir, fusion, new_gray, new_fusion, masked_rgb_img

    eo, fus_y, y, ir, fusion, new_gray, new_fusion, masked_rgb_img = image_process()
    axs[0, 0].imshow(eo[:, :, ::-1])
    axs[0, 0].set_title('eo_' + str(image_num))

    axs[0, 1].imshow(y, cmap='gray')
    axs[0, 1].set_title('eo_y')

    axs[0, 2].imshow(ir, cmap='gray')
    axs[0, 2].set_title('ir_' + str(image_num))

    axs[0, 3].imshow(fus_y, cmap='gray')
    axs[0, 3].set_title('fus_y')

    axs[1, 0].imshow(fusion[:, :, ::-1])
    axs[1, 0].set_title('fusion')

    axs[1, 1].imshow(masked_rgb_img, cmap='gray')
    axs[1, 1].set_title('masked_y')

    axs[1, 2].imshow(new_gray, cmap='gray')
    axs[1, 2].set_title('new_gray_' + str(th))

    axs[1, 3].imshow(new_fusion[:, :, ::-1])
    axs[1, 3].set_title('new_fusion')


# Define the list of images to display

image1 = cv2.imread('D:/new_fusion/895/895_100_eo.png')
image2 = cv2.imread('D:/new_fusion/895/895_100_eo.png')
image3 = cv2.imread('D:/new_fusion/895/895_100_eo.png')
image4 = cv2.imread('D:/new_fusion/895/895_100_eo.png')
image5 = cv2.imread('D:/new_fusion/895/895_100_eo.png')
image6 = cv2.imread('D:/new_fusion/895/895_100_eo.png')
image7 = cv2.imread('D:/new_fusion/895/895_100_eo.png')
image8 = cv2.imread('D:/new_fusion/895/895_100_eo.png')

fig, axs = plt.subplots(2, 4)
img1 = axs[0,0].imshow(image1)
img2 = axs[0,1].imshow(image2)
img3 = axs[0,2].imshow(image3)
img4 = axs[0,3].imshow(image4)

img5 = axs[1,0].imshow(image5)
img6 = axs[1,1].imshow(image6)
img7 = axs[1,2].imshow(image7)
img8 = axs[1,3].imshow(image8)


current_img = img1

def on_press(event):
    global th, image_num
    #global current_img
    global eo, fus_y, y, ir, fusion, new_gray, new_fusion, masked_rgb_img

    print(event.key)
    if event.key == 'right':
        image_num = image_num +1

        print_img(axs)


    elif event.key == 'left':
        image_num = image_num - 1

        print_img(axs)


    elif event.key == 'up':
        th = th + 10
        print_img(axs)


    elif event.key == 'down':
        th = th - 10

        print_img(axs)

    elif event.key == 'x':
        createFolder('D:/new_fusion/' + str(image_num))
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_eo.png', eo)
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_fus_y.png', fus_y)
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_eo_y.png', y)
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_ir.png', ir)
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_fusion.png', fusion)
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_new_gray.png',
                    new_gray)
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_new_fusion.png',
                    new_fusion)
        cv2.imwrite('D:/new_fusion/' + str(image_num) + '/' + str(image_num) + '_' + str(th) + '_masked_y.png',
                    masked_rgb_img)
        plt.savefig('D:/new_fusion/Figure_' + str(image_num) + '_' + str(th) + '.png')


    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_press)
plt.show()