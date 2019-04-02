import skvideo.io
import skvideo.datasets
import skvideo.utils

import scipy.misc
import matplotlib.pyplot as plt
import cv2 








videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
print(videodata.shape)
videodata=skvideo.utils.rgb2gray(videodata)


def callback_left_button(event):
    ''' this function gets called if we hit the left button'''
    print('Left button pressed')


def callback_right_button(event):
    ''' this function gets called if we hit the left button'''
    print('Right button pressed')
    ax.set(xlabel='nuc', ylabel='dfg')
    
fig, ax = plt.subplots()

toolbar_elements = fig.canvas.toolbar.children()
left_button = toolbar_elements[6]
right_button = toolbar_elements[8]

left_button.clicked.connect(callback_left_button)
right_button.clicked.connect(callback_right_button)

ax.imshow(videodata[5,:,:,0], cmap='gray')
ax.set(xlabel='x [px]', ylabel='y [px]')

