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
folder=main_folder+'20_02_25_P3/'
folder=main_folder+'20_02_25_M5/'
#folder=main_folder+'20_02_26_L3/'
#folder=main_folder+'20_02_26_Q3/'
file = 'raw_09_1'



video = Video(folder, file)
video.loadData()


video._video['raw']=video._video['raw'][:,:,:600]
video.refresh()


video.make_diff(10)
video.fouriere()

video.make_toggle(10, 10)
video.explore()

print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
