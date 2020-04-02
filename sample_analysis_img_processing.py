import numpy as np
import time as t

from video_processing import Video
import matplotlib.pyplot as plt


plt.close("all")
time_start=t.time()

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder = main_folder+'20_02_25_P3/'
file = 'raw_12_1'
#file = 'raw_11_1'
#file = 'raw_16_1'


#folder=main_folder+'20_02_26_Q3/'
#file = 'raw_21_1'
#folder=main_folder+'20_03_16_K5/'
#file = 'raw_20_1'
#folder=main_folder+'20_02_26_L3/'
#file = 'raw_03_1'

video = Video(folder, file)
video.loadData()

video._video['raw']=video._video['raw'][100:300,300:500,:]

video.make_diff(k = 10)
video.fouriere(level = 20)

"alpha"

video.img_process_alpha(threshold = 2, noise_level = 0.0012♥)
video.characterize_nps(save = False)
video.exclude_nps([3, 0, 1.1], exclude = True)
video.make_toggle(['diff', 'int'], [10, 10])


"gamma"
#video.load_idea()○
#video.make_corr()
#video.image_process_gamma(threshold = 2.4)  
#video.characterize_nps(save = False)
#video.info_add('\n--auto contrast--')
#video.info_add(video.auto_contrast)
#video.exclude_nps([5, 0, 1.1], exclude = True)
#video.make_toggle(['diff', 'int'], [10, 10])
#video.statistics()

#video.characterize_nps(save = True)


video.statistics()
video.explore()

video.histogram('diff')

print(video.info)
print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
