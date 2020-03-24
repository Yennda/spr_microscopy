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
file = 'raw_05_1'
#file = 'raw_27_1'
#folder=main_folder+'20_02_25_M5/'
#file = 'raw_08_1'

video = Video(folder, file)
video.loadData()




video._video['raw']=video._video['raw'][100:300,300:500,:]
#video._video['raw']=video._video['raw'][109, 84:90, 154:166]            #idea 1
#video._video['raw']=video._video['raw'][125: 131, 100: 110, 103:143]            #idea 2
#video._video['raw']=video._video['raw'][140:170,60:95,200:250]
#video._video['raw']=video._video['raw'][:,:,220:]




video.refresh()
video.make_diff(k = 10)
#video.load_idea('idea_q3_np80')






#video.load_idea('idea_q3_np80')
#video.img_process_alpha(3.5, dip = -0.003, noise_level = 0.001)
#video.image_process_beta(threshold = 100)    #750    
#video.image_process_beta(threshold = 5)     #600

#video.image_process_gamma(threshold = 80)    #750    
#video.save_idea('idea_q3_np80')


#video.make_diff(k = 10)

video.make_toggle(['diff', 'int'], [10, 10])
video.fouriere(level = 20)

#video.characterize_nps()





video.explore_statistics()
video.explore()

#plt.hist(np.matrix.flatten(video.video), 100)

print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))
