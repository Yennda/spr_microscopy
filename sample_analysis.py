from video_processing import VideoLoad
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import tools

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'19_03_13_C3/'
folder=main_folder+'19_03_14_C7/'
folder=main_folder+'19_03_19_C7/'
folder=main_folder+'19_03_28_C6/'


file='meas_diff_02_1'
#file='meas_04_diff_1'


video=VideoLoad(folder,file)
video.loadData()


video.rng=[-0.01, 0.01]
f=20

#video.frame(f)


#video.area_show([20, 280, 1900, 850])
#video.view=[20, 280, 1900, 850]

video.explore()
