from biovideo import BioVideo


import matplotlib.pyplot as plt

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'19_12_04_second_poc/'



file = 'raw_01'


video = BioVideo(folder, file, 2)
video.loadData()
#video.ref_frame=-1
video.makeint()



video.spr=True
video.explore()
