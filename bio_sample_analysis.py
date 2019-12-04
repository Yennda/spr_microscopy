from biovideo import BioVideo


import matplotlib.pyplot as plt

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'2019_11_29_L3_first_experiment/'



file = 'raw_04_2'


video = BioVideo(folder, file)
video.loadData()
video.change_fps(2)

video.rng = [-0.01, 0.01]

#video._video=video._video[:,:,100:]

video.refresh()


video.make_int()
#video.fouriere()
video.explore()
