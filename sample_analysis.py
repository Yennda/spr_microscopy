from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools
import cv2

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'


#folder=main_folder+'20_01_24_third/'
folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'

file = 'raw_01_1'



video = Video(folder, file)
video.loadData()
#video.change_fps(2)

video.rng = [-0.01, 0.01]

#video._video=video._video[:,420:900,:]
#video.refresh()


video.make_int()
#video.fouriere()

video.explore()
