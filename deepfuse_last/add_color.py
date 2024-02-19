import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import keyboard
import pyautogui
from time import sleep
from threading import Thread

loop = 1
th = 100
image_num = 895
image_save = 1   # each image save : 1


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)



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


plt.figure()
plt.subplot(2,4,1)
plt.imshow(eo[:,:,::-1])
plt.title('eo_'+str(image_num))
#cv2.imshow('eo',eo)

plt.subplot(2,4,2)
plt.imshow(y, cmap='gray')
plt.title('eo_y')

plt.subplot(2,4,3)
plt.imshow(ir, cmap='gray')
plt.title('ir_'+str(image_num))
#cv2.imshow('ir',ir)

plt.subplot(2,4,4)
plt.imshow(fus_y, cmap='gray')
plt.title('fus_y')
#cv2.imshow('fus_y',fus_y)

plt.subplot(2,4,5)
plt.imshow(fusion[:,:,::-1])
plt.title('fusion')
#cv2.imshow('fusion',fusion)

plt.subplot(2,4,6)
plt.imshow(masked_rgb_img, cmap='gray')
plt.title('masked_y')
#cv2.imshow('new',masked_rgb_img)

plt.subplot(2,4,7)
plt.imshow(new_gray, cmap='gray')
plt.title('new_gray_'+str(th))

plt.subplot(2,4,8)
plt.imshow(new_fusion[:,:,::-1])
plt.title('new_fusion')

plt.savefig('D:/new_fusion/Figure_'+str(image_num)+'_'+str(th)+'.png')
#cid = plt.connect('keyboard_event',keyboard_event())
plt.show()



