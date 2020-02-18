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

folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'
#folder=main_folder+'20_02_14_P3/'

#file = 'raw_09'
file = 'raw_01'

video = BioVideo(folder, file, 4)
#video.spr = False
video.loadData()

#for vid in video._videos:
#    vid._video['raw'] = vid._video['raw'][600:600+273, :, :]
#    vid.refresh()


#video.ref_frame = 1159
video.ref_frame = 0

#video.make_diff(10)
#video.make_int()
video.make_toggle()
#video.make_both()

#video.fouriere()
video.spr = True
video.spr_std = True


#video._video=video._video[20:150,950:1250,:]
#video.save_array_form()

video.explore()

print('{:.2f} s'.format(t.time()-time_start))



