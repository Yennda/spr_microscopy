from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools
import cv2
from np_analysis import np_analysis, is_np
from image_processing import correlation_temporal
import time as t

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

time_start=t.time()

#folder=main_folder+'20_01_24_third/'

#folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'
#file = 'raw_01_2'

#folder=main_folder+'19_08_29_L3/'
#file = 'raw_32_1'

folder=main_folder+'20_02_18_P3/'
file = 'raw_14_1'


video = Video(folder, file)
video.loadData()


video.rng = [-0.01, 0.01]

#video._video['raw']=video._video['raw'][700+110:700+180,70:130,:370]
video._video['raw']=video._video['raw'][70:220,490:660,:]
#video._video['raw']=video._video['raw'][100:150,500:600,:200]
print('LOAD TIME: {:.2f} s'.format(t.time()-time_start))
#video.change_fps(10)
#video.refresh()


video.make_diff(k = 10)
print('MAKE TIME: {:.2f} s'.format(t.time()-time_start))
#video.fouriere()

# t, y, x
#a
#is_np(video.video[:, 799, 170], show=True)
#correlation_temporal(video.video[:, 799, 170], 10, -0.0055)
#b
#is_np(video.video[:, 739, 218], show=True)
#correlation_temporal(video.video[:, 739, 218], 10, -0.0055)
# a
#is_np(video.video[:, 212, 673], show=True)

#b^
#is_np(video.video[:, 88, 733], show=True)

#c
#is_np(video.video[:, 87, 737], show=True)

video.img_process_alpha(3.5)


video.explore()
print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
