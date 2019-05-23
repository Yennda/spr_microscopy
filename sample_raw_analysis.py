from video_processing import Video
from video_raw import RawVideo
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import tools

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

#folder=main_folder+'19_04_27_B6/'
#folder=main_folder+'19_04_11_C5/'
#folder=main_folder+'19_04_17_C3/'
folder=main_folder+'19_05_09_B6/'
#folder=main_folder+'19_05_15_B3/'

rawfile='meas_raw_05_1'


raw=RawVideo(folder, rawfile)
raw.loadData()

print(np.average(raw.reference[250:1000,:]))



#raw.rng=[-0.01, 0.01]
##raw.rng=[0, 0.4]
#
#print('loaded')
#raw.integrate(3)
#
##raw.reference=raw._video[:,:,0]
##raw.refref()
##raw.refdifffirst()
#raw.refdiff()
#
#raw.explore('diff')
#
##raw.explore('ref')
[0.4245497817708026, 0.4138191904785803, 0.406395374315801, 0.4109129918864613]
[0.5403332568693934, 0.5233846637604816, 0.5140893776139688, 0.5232715346690346]