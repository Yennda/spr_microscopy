from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools
import cv2
from np_analysis import np_analysis, is_np
import time as t

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

time_start=t.time()

#folder=main_folder+'20_01_24_third/'
folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'
folder=main_folder+'19_08_29_L3/'
#folder=main_folder+'20_02_06_Tomas_magnetic_nps/'
folder=main_folder+'20_04_15_L3/'
file = 'raw_17_1'



video = Video(folder, file)
video.loadData()
#video.change_fps(100)

video.rng = [-0.01, 0.01]

video._video['raw']=video._video['raw'][250:350,650:750,:]
video.refresh()


video.make_diff(k = 100)


#video.fouriere()

# t, y, x
#is_np(video.video[:, 799, 170], show=True)

# a
#is_np(video.video[:, 212, 673], show=True)

#b
#is_np(video.video[:, 88, 733], show=True)

video.explore()
print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
