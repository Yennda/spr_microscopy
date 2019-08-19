from video_recognition import VideoRec
import numpy as np

import matplotlib.pyplot as plt
import tools
import cv2

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
#main_folder='C:/SPRUP_data_Jenda/2017_08_09_supercont_NP_imaging/'
folder=main_folder
#folder=main_folder+'19_04_11_C5/'  #low noise
#folder=main_folder+'19_04_17_C3/'  #low noise
#folder=main_folder+'19_05_09_B6/'
#folder=main_folder+'19_05_15_B3/'   #high noise
folder=main_folder+'19_07_18_C5/'
#folder=main_folder+'19_03_28_C6/'
folder=main_folder+'19_08_16_K4/'
#folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'


#file='diff_02_1'
file='norm_05_1'
#file='meas_diff_03_1'
#file='R_184_l_182_NP_150_1000x_raw_1'




video=VideoRec(folder, file)
video.loadData()
video.rng=[-0.01, 0.01]




video.fouriere()
#video._video=video._video[40:160, 750:1000,:]
#mask=video.np_recognition()
#video.explore()
#video.area_show([20, 280, 1900, 850])
#video.view=[20, 280, 1900, 850]


mask=mask.astype(np.uint8)
#image = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
#cvuint8 = cv2.convertScaleAbs(image)

im_bw = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
(tresh, im_bw) = cv2.threshold(im_bw, 100, 255, 0)

#th, threshed = cv2.threshold(image, 100, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## findcontours
cnts = cv2.findContours(tresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

s1= 3
s2 = 20
xcnts = []
for cnt in cnts:
    if s1<cv2.contourArea(cnt) <s2:
        xcnts.append(cnt)

print("Dots number: {}".format(len(xcnts)))


plt.subplot(211),plt.imshow(video._video[:,:,-1], cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(mask, cmap = 'gray')
plt.title('Recognized NPs'), plt.xticks([]), plt.yticks([])
plt.show()