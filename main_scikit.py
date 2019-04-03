import skvideo.io
import skvideo.datasets
import skvideo.utils

import scipy.misc
import matplotlib.pyplot as plt
import cv2 

plt.close('all')






videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
print(videodata.shape)
videodata=skvideo.utils.rgb2gray(videodata)
videodata=videodata[:,:,:,0]


fig2 = plt.figure()
ax2=fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.imshow(videodata[5,:,:], cmap='gray')
ax2.set_title('old')
ax2.set(xlabel='x [px]', ylabel='y [px]')


#Mouse scroll event.
def mouse_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'down':
        next_slice(ax)
    elif event.button == 'up':
        prev_slice(ax)
    fig.canvas.draw()

#Next slice func.
def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]
    img.set_array(volume[ax.index])

def prev_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    img.set_array(volume[ax.index])

def mouse_click(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    volume = videodata
    ax.volume = volume
    ax.index = (ax.index - 1) % volume.shape[0]              
    img.set_array(volume[ax.index])
    fig.canvas.draw_idle()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    volume = videodata
    ax.volume = volume
    ax.index = 1
    img = ax.imshow(volume[ax.index], cmap='gray')
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    plt.show()