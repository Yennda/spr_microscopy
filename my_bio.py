from biovideo import BioVideo
import matplotlib.pyplot as plt
import tools as tl
import time as t

time_start=t.time()

tl.clear_all()
plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder=main_folder+'19_11_29_L3_first_experiment/'
folder=main_folder+'19_12_04_second_poc/'
folder=main_folder+'20_01_24_third/'

#folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'
#folder=main_folder+'20_02_05_Tomas_distance_from_surface/'
#folder=main_folder+'20_02_18_P3/'
#folder=main_folder+'20_03_13_Q3/'
folder=main_folder+'20_04_21_L3_tomas/'

#file = 'raw_09'
file = 'raw_08'

video = BioVideo(folder, file, 3)
#video.spr = False

video.loadData()

#for vid in video._videos:
#    vid._video['raw'] = vid._video['raw'][:, 400:800, :100*10]
#    vid._video['raw'] = vid._video['raw'][:, 400:800, :100*10]
#    vid.refresh()


#video.ref_frame = 1159
video.ref_frame = 0

#video.make_diff(10)
#video.make_int(10)
video.make_toggle(kd=10, ki=10)
#video.make_both()

#video.fouriere()
video.spr = True
video.spr_std = True


#video._video=video._video[20:150,950:1250,:]
#video.save_array_form()

video.explore()

print('{:.2f} s'.format(t.time()-time_start))



