import numpy as np
import time as t

from video_processing import Video
import matplotlib.pyplot as plt


#plt.close("all")
time_start=t.time()

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder=main_folder+'20_02_25_P3/'
file = 'raw_10_1'
#file = 'raw_11_1'
#file = 'raw_16_1'


folder=main_folder+'20_02_26_Q3/'
file = 'raw_19_1'

#folder=main_folder+'20_02_26_L3/'
#file = 'raw_03_1'

video = Video(folder, file)
video.loadData()

video._video['raw']=video._video['raw'][100:300,300:500,:]

video.make_diff(k = 10)
video.fouriere(level = 20)

#video.load_idea()

video.load_idea('idea_q3_04_np80')
#video.load_idea('idea_q3_09_np80')

#video.make_corr()

video.image_process_gamma(threshold = 52)  
video.characterize_nps(save = False)
video.exclude_nps([2], exclude = True)
video.statistics()


#video.characterize_nps(save = True)

video.make_toggle(['diff', 'corr'], [10, 10])

video.explore()

video.histogram()

print(video.info)
print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
