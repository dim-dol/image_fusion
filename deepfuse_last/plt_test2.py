import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import keyboard
import pyautogui
from time import sleep
from threading import Thread

loop = 1
th = 120
image_num = 895
image_save = 1   # each image save : 1

global eo, ir, fus_y, y, fusion, new_gray, new_fusion, masked_rgb_img

def keyboard_event(event):
    global loop, th, image_num, image_save
    global eo, ir, fus_y, y, fusion, new_gray, new_fusion, masked_rgb_img

    print('press', event.key)
    #print(keyboard.read_key())
    #sleep(0.01)
    if event.key == 80:
        # 아래쪽 방향키
        th = th - 1
    elif event.key == 72:
        # 위쪽 방향키
        th = th + 1
    elif event.key =='right':
        image_num = image_num + 1
    elif event.key =='left':
        image_num = image_num - 1
    elif event.key =='enter':
        image_save = 1
    elif event.key =='x':
        loop = 0

    image_process()

    image_save_func()
    plt.figure()
    plt.subplot(2, 4, 1)
    plt.draw(eo[:, :, ::-1])
    plt.title('eo_' + str(image_num))
    # cv2.imshow('eo',eo)

    plt.subplot(2, 4, 2)
    plt.draw(y, cmap='gray')
    plt.title('eo_y')

    plt.subplot(2, 4, 3)
    plt.draw(ir, cmap='gray')
    plt.title('ir_' + str(image_num))
    # cv2.imshow('ir',ir)

    plt.subplot(2, 4, 4)
    plt.draw(fus_y, cmap='gray')
    plt.title('fus_y')
    # cv2.imshow('fus_y',fus_y)

    plt.subplot(2, 4, 5)
    plt.draw(fusion[:, :, ::-1])
    plt.title('fusion')
    # cv2.imshow('fusion',fusion)

    plt.subplot(2, 4, 6)
    plt.draw(masked_rgb_img, cmap='gray')
    plt.title('masked_y')
    # cv2.imshow('new',masked_rgb_img)

    plt.subplot(2, 4, 7)
    plt.draw(new_gray, cmap='gray')
    plt.title('new_gray_' + str(th))

    plt.subplot(2, 4, 8)
    plt.draw(new_fusion[:, :, ::-1])
    plt.title('new_fusion')

    #plt.savefig('D:/new_fusion/Figure_' + str(image_num) + '_' + str(th) + '.png')

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)



def image_process():
    global loop, th, image_num, image_save
    global eo, ir, fus_y, y, fusion, new_gray, new_fusion, masked_rgb_img
    #new_gray = np.zeros((480,640))
    fus_y = cv2.imread('./outputs/fused'+str(image_num)+'_deepfuse_addition1.png',cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread('../dataset/dataset_eoir/IR/IR_'+str(image_num)+'.png',cv2.IMREAD_GRAYSCALE)
    eo = cv2.imread('../dataset/dataset_eoir/EO/EO_'+str(image_num)+'.png',cv2.IMREAD_COLOR)

    color_ycbcr = cv2.cvtColor(eo, cv2.COLOR_BGR2YCrCb)

    y, cb, cr = cv2.split(color_ycbcr[:,:,0:3])

    # Apply thresholding to the infrared image
    infrared_mask = cv2.threshold(ir, th, 255, cv2.THRESH_BINARY)[1]

    # Apply the mask to the RGB image
    masked_rgb_img = cv2.bitwise_and(ir, fus_y, mask=infrared_mask)
    #masked_rgb_img=masked_rgb_img*1.2
    new_gray = masked_rgb_img

    for i in range(fus_y.shape[0]):
        for j in range(fus_y.shape[1]):
            if masked_rgb_img[i,j] == 0:
                new_gray[i, j] = y[i,j]

    fusion = cv2.merge((fus_y,cb,cr))
    fusion = cv2.cvtColor(fusion,cv2.COLOR_YCrCb2BGR)

    new_fusion = cv2.merge((masked_rgb_img,cb,cr))
    new_fusion = cv2.cvtColor(new_fusion,cv2.COLOR_YCrCb2BGR)

def image_save_func():
    global loop, th, image_num, image_save
    global eo, ir, fus_y, y, fusion, new_gray, new_fusion, masked_rgb_img

    if image_save == 1:
        createFolder('D:/new_fusion/'+str(image_num))
        cv2.imwrite('D:/new_fusion/'+str(image_num)+'/'+str(image_num)+'_'+str(th)+'_eo.png',eo)
        cv2.imwrite('D:/new_fusion/'+str(image_num)+'/' + str(image_num) + '_' + str(th) + '_fus_y.png', fus_y)
        cv2.imwrite('D:/new_fusion/' +str(image_num)+'/'+ str(image_num) + '_' + str(th) + '_eo_y.png', y)
        cv2.imwrite('D:/new_fusion/'+str(image_num)+'/' + str(image_num) + '_' + str(th) + '_ir.png', ir)
        cv2.imwrite('D:/new_fusion/'+str(image_num)+'/' + str(image_num) + '_' + str(th) + '_fusion.png', fusion)
        cv2.imwrite('D:/new_fusion/'+str(image_num)+'/' + str(image_num) + '_' + str(th) + '_new_gray.png', new_gray)
        cv2.imwrite('D:/new_fusion/'+str(image_num)+'/' + str(image_num) + '_' + str(th) + '_new_fusion.png', new_fusion)
        cv2.imwrite('D:/new_fusion/'+str(image_num)+'/' + str(image_num) + '_' + str(th) + '_masked_y.png', masked_rgb_img)


        image_save = 0

fig, ax = plt.subplots(2,4)

#image_process()

#plt.figure()

# plt.subplot(2,4,1)
# plt.imshow(eo[:,:,::-1])
# plt.title('eo_'+str(image_num))
# #cv2.imshow('eo',eo)
#
# plt.subplot(2,4,2)
# plt.imshow(y, cmap='gray')
# plt.title('eo_y')
#
# plt.subplot(2,4,3)
# plt.imshow(ir, cmap='gray')
# plt.title('ir_'+str(image_num))
# #cv2.imshow('ir',ir)
#
# plt.subplot(2,4,4)
# plt.imshow(fus_y, cmap='gray')
# plt.title('fus_y')
# #cv2.imshow('fus_y',fus_y)
#
# plt.subplot(2,4,5)
# plt.imshow(fusion[:,:,::-1])
# plt.title('fusion')
# #cv2.imshow('fusion',fusion)
#
# plt.subplot(2,4,6)
# plt.imshow(masked_rgb_img, cmap='gray')
# plt.title('masked_y')
# #cv2.imshow('new',masked_rgb_img)
#
# plt.subplot(2,4,7)
# plt.imshow(new_gray, cmap='gray')
# plt.title('new_gray_'+str(th))
#
# plt.subplot(2,4,8)
# plt.imshow(new_fusion[:,:,::-1])
# plt.title('new_fusion')

#plt.savefig('D:/new_fusion/Figure_'+str(image_num)+'_'+str(th)+'.png')
fig.canvas.mpl_connect('key_press_event',keyboard_event)
plt.show()



'''

import cv2
import numpy as np

img = cv2.imread('D:/EO_2.bmp', cv2.IMREAD_COLOR)

img_resized = cv2.resize(img, (640, 480), cv2.INTER_AREA)
img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2Lab)
cv2.imwrite('D:/EO_2.png', img_normalized)

img = cv2.imread('D:/EO_2.png', cv2.IMREAD_UNCHANGED)

cv2.imshow('ddd',img)
cv2.waitKey(0)
#img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

#img_rgb = img_resized.astype(np.float32) / 255.0

#cv2.imwrite('D:/EO_2.png',(img_rgb * 255).astype(np.uint8))r
'''