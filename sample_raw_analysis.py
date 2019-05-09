from video_processing import Video
from video_raw import RawVideo
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import tools

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

#folder=main_folder+'19_04_27_B6/'
folder=main_folder+'19_04_11_C5/'
#folder=main_folder+'19_04_17_C3/'


rawfile='meas_raw_07_1'


raw=RawVideo(folder, rawfile)
raw.loadData()
raw.rng=[-0.01, 0.01]


print('loaded')
#raw.reference=raw._video[:,:,0]
#raw.refref()
raw.refdifffirst()
#raw.refdiff()

raw.explore('diff')

#raw.explore('ref')
