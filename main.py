from video_processing import VideoLoad
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
import tools

plt.close("all")

folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/19_03_19_C5/'
file='meas_01_diff_1'


video=VideoLoad(folder+file)
video.loadData()


video.rng=[-0.01, 0.01]
f=20

#video.frame(f)


#video.area_show([20, 280, 1900, 850])
video.view=[20, 280, 1900, 850]
video.explore()

#
#fr=[0,20]
#video_load=video.video[video.view[1]:video.view[1]+video.view[3], video.view[0]:video.view[0]+video.view[2], fr[0]:fr[1]]
#fig = plt.figure()
#data = video_load[:,:,0]
#
#im = plt.imshow(data, cmap='gray', vmin=video.rng[0], vmax=video.rng[1], animated=True)
#def init():
#    im.set_data(data)
#
#def animate(i):
#    plt.title(str(i)+'/'+str(fr[1]-fr[0]))
#    im.set_data(video_load[:,:,i])
#    return im
#
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=fr[1]-fr[0],
#                               interval=50)
#plt.show()
#
