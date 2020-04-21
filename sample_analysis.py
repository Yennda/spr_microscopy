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
folder=main_folder+'20_02_14_P3/'
folder=main_folder+'20_03_11_prism/'
folder=main_folder+'20_03_18_flat_gold/'
#file = 'raw_32_1'
file = 'raw_01_1'


video = Video(folder, file)
video.loadData()
#video.change_fps(20)


video.rng = [-0.01, 0.01]

#video._video['raw']=video._video['raw'][:,400:800,:]
video.refresh()


#video.make_diff(k = 10)
#video.make_int()
#video.fouriere()


video.explore()
print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
