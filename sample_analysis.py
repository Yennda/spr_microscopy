from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools

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
#folder=main_folder+'19_07_16_ultraplacad/'
#folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'


#file='diff_02_1'
file='neref_02_1'
#file='meas_diff_03_1'
#file='R_184_l_182_NP_150_1000x_raw_1'




video=Video(folder, file)
video.loadData()
video.rng=[-0.01, 0.01]
video.explore()
#video.area_show([20, 280, 1900, 850])
#video.view=[20, 280, 1900, 850]
print(video._video.shape)
