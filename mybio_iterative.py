# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:42:26 2020

@author: bukacek
"""

from biovideo import BioVideo
import matplotlib.pyplot as plt
import tools as tl
import time as t

time_start=t.time()

tl.clear_all()
plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
#folder=main_folder+'19_12_04_second_poc/'
#folder=main_folder+'20_01_24_third/'

#folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'
folder=main_folder+'20_02_18_P3/'
folder=main_folder+'20_03_13_Q3/'
#file = 'raw_09'
file = 'raw_03'

video = BioVideo(folder, file, 1)
#video.spr = False

f = open(folder + 'info.txt', "a+", encoding="utf-8")
  

#            plt.close(2)
plt.close(3)
for i in range(35, 36):
    lm = 700 + i
    start = 261 + i*229 + 40
    end = 261 + (i + 1)*229-40
    
    print('lm: {}'.format(lm))
    print('start: {}'.format(start))
    print('end: {}'.format(end))
    video.loadData()
    
    for vid in video._videos:
        vid._video['raw'] = vid._video['raw'][:, 400:800, start:end]
        vid.refresh()
    
    video.ref_frame = 0
    video.make_int(10)


    video.spr = True
    video.spr_std = True

    time, spr, std = video.explore(lm))
    f.write('\n')
#    plt.close("all")
    
#    print('{:.2f} s'.format(t.time()-time_start))
f.close()


