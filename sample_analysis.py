from video_processing import Video
import numpy as np

import matplotlib.pyplot as plt
import tools
import cv2

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'


#folder=main_folder+'20_01_24_third/'
folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'
#folder=main_folder+'19_09_13_M5/'
#folder=main_folder+'19_09_05_M5/'
file = 'raw_01_2'



video = Video(folder, file)
video.loadData()
#video.change_fps(2)

video.rng = [-0.01, 0.01]

#video._video['raw']=video._video['raw'][300:800,:,:]
#video.refresh()


video.make_diff(k = 5)
#video.fouriere()

video.explore()
