from biovideo import BioVideo
import matplotlib.pyplot as plt
import tools as tl

tl.clear_all()
plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
#folder=main_folder+'19_12_04_second_poc/'
folder=main_folder+'20_01_24_third/'


file = 'raw_01'
#file = 'raw_03'

video = BioVideo(folder, file, 1)
video._channels=[0]
video.loadData()
#video.ref_frame=-1
video.make_toggle()
#video.make_diff()
#video.make_both()

#video.ref_frame = 0



#video._video=video._video[20:150,950:1250,:]


video.spr = False
video.explore()
