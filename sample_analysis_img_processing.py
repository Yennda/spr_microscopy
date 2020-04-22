import numpy as np
import time as t

from video_processing import Video
import matplotlib.pyplot as plt


plt.close("all")
time_start=t.time()

def change(video, thresholds):
    video.exclude_nps(thresholds, exclude = True)
    video.statistics()
    video.explore()
    
main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder = main_folder+'20_02_25_P3/'
file = 'raw_20_1'

#folder = main_folder+'20_02_26_Q3/'
#file = 'raw_15_1'

folder=main_folder+'20_03_16_K5/'
file = 'raw_18_1'

#folder=main_folder+'20_02_26_L3/'
#file = 'raw_05_1'

#folder=main_folder+'20_01_24_third/'
#file = 'raw_01_1'
#
#folder=main_folder+'20_02_25_M5/'
#file = 'raw_08_1'

#folder=main_folder+'20_01_24_third/'
#file = 'raw_01_1'

#folder=main_folder+'20_04_03_L3/'
#file = 'raw_07_1'

#folder=main_folder+'20_03_23_L3_4x/'
#file = 'raw_06_1'

#folder=main_folder+'20_04_14_M5/'
#file = 'raw_16_1'

#folder=main_folder+'20_04_15_L3/'
#file = 'raw_07_1'
#
folder=main_folder+'20_04_20_Q4/'
file = 'raw_15_1'


video = Video(folder, file)
video.loadData()

video._video['raw']=video._video['raw'][100:,430:730,:]
video.make_diff(k = 10)
#video.change_fps(10)
#video.make_int(k = 10)
video.fouriere(level = 20)

"alpha"

#video.img_process_alpha(threshold = 2.5, noise_level = 0.0012U)
#video.characterize_nps(save = False)
#video.exclude_nps([3], exclude = True)
#video.make_toggle(['diff', 'int'], [10, 10])

"gamma"
video.load_idea()
video.make_corr()
video.image_process_gamma()  
video.characterize_nps(save = False)
#video.info_add('\n--auto contrast--')
#video.info_add(video.auto_contrast)
video.exclude_nps([2], exclude = True)
video.make_toggle(['diff', 'corr'], [10, 10])

video.statistics()

#video.characterize_nps(save = True)


video.explore()
video.histogram('corr')

#video.save_info_measurement(80, 734)

video.info_add('\n--elapsed time--\n{:.2f} s'.format(t.time()-time_start))
print(video.info)