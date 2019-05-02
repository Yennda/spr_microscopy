from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder=main_folder+'19_04_27_B6/'
folder=main_folder+'19_04_11_C5/'
#folder=main_folder+'19_04_17_C3/'

file='meas_diff_05_1'


video=Video(folder, file)
video.loadData()
video.rng=[-0.01, 0.01]
video.explore()
#video.area_show([20, 280, 1900, 850])
#video.view=[20, 280, 1900, 850]
