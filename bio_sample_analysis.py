from biovideo import BioVideo
import matplotlib.pyplot as plt

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'19_12_04_second_poc/'
folder=main_folder+'20_01_24_third/'


file = 'raw_01'


video = BioVideo(folder, file, 1)
video.loadData()
#video.ref_frame=-1
#video.makediff()

#video._video=video._video[20:150,950:1250,:]


video.spr=True
video.explore()
