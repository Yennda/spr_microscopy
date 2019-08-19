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
#folder=main_folder+'19_08_16_K4/'
#folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'


#file='diff_02_1'
#file='norm_05_1'
file='neref_02_1'
#file='R_184_l_182_NP_150_1000x_raw_1'




video=VideoRec(folder, file)
video.loadData()
video.rng=[-0.01, 0.01]




video.fouriere()
video._video=video._video[75:220, 1100:1400,:]
#mask=video.np_recognition()
video.explore()

#
#gray = mask.astype(np.uint8)
#th, threshed = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
#s1= 1
#s2 = 20
#xcnts = []
#control_mask=np.zeros(mask.shape)
#for cnt in cnts:
#    if s1<cv2.contourArea(cnt) <s2:
#        
#        for c in cnt:
#            control_mask[c[0][1], c[0][0]]=1
#        xcnts.append(cnt)
#
#print("Dots number: {}".format(len(xcnts)))
#
#
#
#video_dark=(video._video[:,:,-1]+1e-02)*1e04
#
#video_rgb=cv2.cvtColor(video_dark.astype(np.uint8), cv2.COLOR_GRAY2RGB)
#video_rgb[:,:,0]=control_mask*255
#
#plt.imshow(video_rgb)




#plt.subplot(311),plt.imshow(video._video[:,:,-1], cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(312),plt.imshow(mask, cmap = 'gray')
#plt.title('Recognized pixels'), plt.xticks([]), plt.yticks([])
#plt.subplot(313),plt.imshow(control_mask, cmap = 'gray')
#plt.title('Recognized NPs'), plt.xticks([]), plt.yticks([])
#plt.show()