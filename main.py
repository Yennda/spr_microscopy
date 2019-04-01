from video_processing import VideoLoad
import numpy as np
import cv2
import time

plt.close("all")

folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/19_03_19_C5/'
file='meas_01_ diff_1'

video=VideoLoad(folder+file)
video.loadData()


video.rng=[-0.01, 0.01]
f=20



video.frame(f)


#video.area_show([20, 280, 1900, 850])
video.view=[20, 280, 1900, 850]
video.frame(20)


video.play([0,100], 0.1)
