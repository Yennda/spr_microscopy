import numpy as np
import time as t

from video_processing import Video
import matplotlib.pyplot as plt


plt.close("all")
time_start=t.time()

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder=main_folder+'20_02_25_P3/'
file = 'raw_10_1'
#file = 'raw_11_1'
#file = 'raw_16_1'


folder=main_folder+'20_02_26_Q3/'
file = 'raw_18_1'

#folder=main_folder+'20_02_26_L3/'
#file = 'raw_03_1'

video = Video(folder, file)
video.loadData()

video._video['raw']=video._video['raw'][100:300,300:500,:]

video.make_diff(k = 10)
video.fouriere(level = 20)


video.load_idea('idea_q3_04_np80')
video.make_corr()

#video.img_process_alpha(3.5, dip = -0.003, noise_level = 0.001)
#video.image_process_beta(threshold = 100)    #750    
video.image_process_gamma(threshold = 60)    #750    


video.characterize_nps(save = True)
video.exclude_nps(contrast = 2)
video.statistics()

video.make_toggle(['diff', 'corr'], [10, 10])

video.explore()

#video.make_corr()
#video.histogram()
#plt.hist(np.matrix.flatten(video.video), 100)

#print(video.np_count_integral[-1])
print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
