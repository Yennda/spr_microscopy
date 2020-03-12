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

#folder=main_folder+'20_02_18_P3/'
#file = 'raw_14_1'

folder=main_folder+'20_02_25_P3/'
file = 'raw_10_1'
#file = 'raw_11_1'
#file = 'raw_16_1'

#file = 'raw_05_1'
folder=main_folder+'20_02_26_Q3/'
file = 'raw_04_1'
#file = 'raw_27_1'
#folder=main_folder+'20_02_25_M5/'
#file = 'raw_08_1'

video = Video(folder, file)
video.loadData()


#video._video['raw']=video._video['raw'][700+110:700+180,70:130,:370]
#video._video['raw']=video._video['raw'][70:220,490:660,:]
#video._video['raw']=video._video['raw'][100:150,500:600,:200]

video._video['raw']=video._video['raw'][100:300,300:500,:300]
#video._video['raw']=video._video['raw'][109, 84:90, 154:166]            #idea 1
#video._video['raw']=video._video['raw'][125: 131, 100: 110, 103:143]            #idea 2
#video._video['raw']=video._video['raw'][140:170,60:95,200:250]
#video._video['raw']=video._video['raw'][:,:,220:]


#video._video['raw']=video._video['raw'][100:150,300:400,100:300]
#video._video['raw']=video._video['raw'][100:140,300:340,200:250]

print('LOAD TIME: {:.2f} s'.format(t.time()-time_start))
#video.change_fps(10)+
video.refresh()


video.make_diff(k = 10)

# t, y, x
#a
#correlation_temporal(video.video[:, 115, 116], k_diff=10, show=True)

print('MAKE TIME: {:.2f} s'.format(t.time()-time_start))
#video.fouriere(level = 20)

#video.img_process_alpha(3.5, dip = -0.003, noise_level = 0.001)

time_beta = t.time()
#video.image_process_beta(threshold = 60)    #750    
#video.image_process_beta(threshold = 5)     #600

video.image_process_gamma(threshold = 75)    #750    
print('CORRELATION TIME: {:.2f} s'.format(t.time()-time_beta))



#video.make_frame_stats()
#video.ref_frame = 20
#video.make_toggle(10, 10)
video.recognition_statistics()
#video.explore()

#plt.hist(np.matrix.flatten(video.video), 100)

print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
