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
#folder=main_folder+'20_02_25_P3/'
folder=main_folder+'20_02_25_M5/'
#folder=main_folder+'20_02_26_L3/'
#folder=main_folder+'20_02_26_Q3/'

stds = []
amplitude = []

for i in range(1, 17):
    plt.close("all")
    
    file = 'raw_{:02d}_1'.format(i)
    
    
    
    video = Video(folder, file)
    video.loadData()
    
    
    video._video['raw']=video._video['raw'][:,400:800,:200]
    video.refresh()
    
    
    video.make_int(10)
    amplitude.append(video.fouriere(20, show=True))
    
    #video.make_toggle(10, 10)
#    stds.append(np.std(video.video[-20:-1,:,:]))
    stds.append(np.average(np.abs((video.video[-20:-1,:,:]))))
    video.show_stats = True
    
    video.explore()
#    plt.hist(np.matrix.flatten(video.video[-2, :, :]), 100)
    
print(amplitude)
print(stds)    
print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
