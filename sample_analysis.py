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
#folder=main_folder+'19_09_13_M5/'
#folder=main_folder+'20_02_06_Tomas_magnetic_nps/'
file = 'raw_01_2'



video = Video(folder, file)
video.loadData()
#video.change_fps(2)

video.rng = [-0.01, 0.01]

video._video['raw']=video._video['raw'][:,:,:600]
video.refresh()


video.make_int(k = 20)
video.fouriere()

#is_np(video.video[:, 799, 170], show=True)

#video.explore()
print('{:.2f} s'.format(t.time()-time_start))
