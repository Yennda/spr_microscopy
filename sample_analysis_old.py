from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools
import cv2

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
#main_folder='C:/SPRUP_data_Jenda/2020_01_23_Jenda_prism_based_SPRM/'
# main_folder='C:/SPRUP_data_Jenda/2017_08_09_supercont_NP_imaging/'
# folder=main_folder
#folder=main_folder+'19_04_11_C5/'  #low noise

#folder=main_folder+'19_04_27_B6/'
#folder=main_folder+'19_05_09_B6/'
#folder=main_folder+'19_05_15_B3/'   #high noise
#folder=main_folder+'19_07_18_C5/'
#folder=main_folder+'19_08_16_K4/'
#folder=main_folder+'19_08_16_L3/'
#folder=main_folder+'19_08_29_L3/'
#folder=main_folder+'19_09_02_L3/'
#folder=main_folder+'19_09_04_M4/'
#folder=main_folder+'19_09_05_M5/'
#folder=main_folder+'19_09_06_K4/'
#folder=main_folder+'19_09_10_K4/'
#folder=main_folder+'19_09_11_L3_II_SEM/'
#folder=main_folder+'19_09_13_L3_iv_SEM/'
#folder=main_folder+'19_09_13_M5/'
#folder=main_folder+'19_10_30_L3/'
folder=main_folder+'19_10_30_K4/'
#folder=main_folder+'20_01_24_third/'


#folder='C:/SPRUP_data_Jenda/camera_test/'



file = 'raw_01_1'



video = Video(folder, file)
video.loadData()
#video.change_fps(2)

video.rng = [-0.01, 0.01]
#video._video=video._video[20:150,950:1250,:]
#video._video=video._video[20:150,950:1250,:]
#video._video=video._video[:,420:900,:]

#video._video['raw']=video._video['raw'][:,:,:]


#video._video=video._video[500:500+273,200:1664+200,:]
video.refresh()


video.make_int()
#video.fouriere()
video.explore()

#mask=video.np_pixels(inten_a=1e-03, inten_b=2e-4)
#video.np_count(mask, s1=5, s2=20, show=True)


# plt.imshow(mask)a