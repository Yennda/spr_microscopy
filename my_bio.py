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


file = 'raw_01'
#file = 'raw_03'

video = BioVideo(folder, file, 2)
video.loadData()


video.ref_frame = 1159
#video.ref_frame = 0

video.make_diff()
#video.make_int()
#video.make_toggle()
#video.make_both()

#video.fouriere()
video.spr = True


#video._video=video._video[20:150,950:1250,:]
#video.save_array_form()

video.explore()

print('{:.2f} s'.format(t.time()-time_start))