from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools
import cv2

plt.close("all")

# main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
# main_folder='C:/SPRUP_data_Jenda/2017_08_09_supercont_NP_imaging/'
# folder=main_folder
# folder=main_folder+'19_04_11_C5/'  #low noise
# folder=main_folder+'19_04_17_C3/'  #low noise
# folder=main_folder+'19_05_09_B6/'
# folder=main_folder+'19_05_15_B3/'   #high noise
# folder=main_folder+'19_07_18_C5/'
# folder=main_folder+'19_03_28_C6/'
# folder=main_folder+'19_07_16_ultraplacad/'
folder = 'C:/Users/jabuk/Documents/jaderka/ufe/data/'

# file='diff_02_1'
file = 'norm_05_1'
# file='meas_diff_03_1'
# file='R_184_l_182_NP_150_1000x_raw_1'


video = Video(folder, file)
video.loadData()
video.rng = [-0.01, 0.01]

video.fouriere()
# video._video=video._video[180:220, 1100:1200,:]


video.explore()
# mask=video.np_pixels(inten_a=1e-04, inten_b=5e-4)
# video.np_count(mask, s1=2, s2=20, show=True)

# video.explore()
# plt.imshow(mask)

##


print(video.video_stats)
