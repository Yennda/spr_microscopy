from biovideo import BioVideo
import matplotlib.pyplot as plt
import tools as tl
import time as t

time_start=t.time()

tl.clear_all()
#plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
#folder=main_folder+'19_12_04_second_poc/'
#folder=main_folder+'20_01_24_third/'

folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'
file = 'raw_01'
#file = 'raw_03'

folder=main_folder+'20_02_18_P3/'
file = 'raw_14'


video = BioVideo(folder, file, 1)
video.loadData()

for vid in video._videos:
#    vid._video['raw'] = vid._video['raw'][600:600+273, :, :]
#    vid._video['raw'] = vid._video['raw'][:, :, :600]
    vid._video['raw']=vid._video['raw'][100:150,500:600,:200]
    
    vid.refresh()
#video.change_fps(10)

#video.ref_frame = 1159
video.ref_frame = 0

#video.make_diff()
#video.make_int(20)
video.make_toggle(kd=10, ki=10)
#video.make_both()

#video.fouriere()▼
video.spr = True
video.spr_std = True


#video._video=video._video[20:150,950:1250,:]
#video.save_array_form()

video.explore()

print('{:.2f} s'.format(t.time()-time_start))



