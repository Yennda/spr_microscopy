from video_processing import Video
from video_raw import RawVideo
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import tools

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'19_03_13_C3/'
#folder=main_folder+'19_03_14_C7/'
#folder=main_folder+'19_03_19_C7/'
#folder=main_folder+'19_03_28_C6/'

file='meas_01_norm_diff_1'
#file='meas_diff_03_1'
#file='meas_02_diff_1'
#file='meas_07_diff_1'
#
#video=Video(folder,file)
#video.loadData()
#video.rng=[-0.01, 0.01]
#video.explore()


#video.area_show([20, 280, 1900, 850])
#video.view=[20, 280, 1900, 850]

rawfile='meas_01_raw_1'
raw=RawVideo(folder, rawfile)
raw.loadData()
raw.show(raw.reference)
