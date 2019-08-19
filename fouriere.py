import cv2
import numpy as np
from matplotlib import pyplot as plt
from video_processing import Video

#folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'
main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'19_08_16_K4/'
#vyska, sirka, cas
file='norm_05_1'


video=Video(folder, file)
video.loadData()


#img = cv2.imread('AR_images/stinkbug.png',0)
img=video._video[:,:,-1]


f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))



#fshift[:, 460:530] = 0
#fshift[:, 1120:1180] = 0

fshift[:, :530] = 0
fshift[:, 1120:] = 0

#rows, cols = img.shape
#crow,ccol = int(rows/2) , int(cols/2)
#
#mask=np.zeros(img.shape)
#mask[crow-42:crow+42, ccol-314:ccol+314] = 1
#mask_bool=mask==0
#
#fshift[mask_bool]=0
f_ishift = np.fft.ifftshift(fshift)
magnitude_spectrum_filtered = 20*np.log(np.abs(fshift))

img_back = np.fft.ifft2(f_ishift)
img_back = np.real(img_back)

#plt.subplot(211),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(212),plt.imshow(img_back, cmap = 'gray')
#plt.title('Filtered low frequencies'), plt.xticks([]), plt.yticks([])
#plt.show()

plt.subplot(411),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(412),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(413),plt.imshow(magnitude_spectrum_filtered, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(414),plt.imshow(img_back, cmap = 'gray')
plt.title('Filtered low frequencies'), plt.xticks([]), plt.yticks([])
plt.show()
