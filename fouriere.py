import cv2
import numpy as np
from matplotlib import pyplot as plt
from video_processing import Video

folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'
#vyska, sirka, cas
file='norm_05_1'


video=Video(folder, file)
video.loadData()


img = cv2.imread('messi5.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()