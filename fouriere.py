import cv2
import numpy as np
import math as m
from matplotlib import pyplot as plt
from video_processing import Video
import copy
import time as t

time_start = t.time()

# https://lmfit.github.io/lmfit-py/model.html
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

plt.close('all')

# folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'
main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder = main_folder + '19_08_16_K4/'
folder = main_folder + '20_01_24_third/'
folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'

folder=main_folder+'20_02_25_P3/'
file = 'raw_11_1'

#[200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
#[1, 2, 5, 10, 20, 30, 40, 50]





appendix = 'processing'


# folder=main_folder+'19_07_18_C5/'
# file='neref_02_1'
#file = 'norm_05_1'
#file = 'raw_01_2'


for fr in [288]:
    
    frame = fr
    
    video = Video(folder, file)
    video.loadData()
    video._video['raw']=video._video['raw'][100:300,300:500,:300]
#    video.ref_frame = 1159U
    video.make_diff(10)
    img = video._video['diff'][:, :, frame]
    img = img.T
        
    for tr in [1]:
        threshold = tr
        # img = cv2.imread('AR_images/stinkbug.png',0)
    
        
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        
        Lm = 82/20.0
        theta = m.pi*39.5/180
        
        
        k = [m.sin(theta)/Lm*img.shape[0], m.cos(theta)/Lm*img.shape[1]]
        

        
        coor = []
        coor.append([int(img.shape[i]/2+k[i]) for i in range(2)])
        coor.append([int(img.shape[i]/2-k[i]) for i in range(2)])
        
        size = 7
        
        mask = np.full(magnitude_spectrum.shape, False, dtype=bool)
        for c in coor:
            mask[c[0]-size:c[0]+size, c[1]-size:c[1]+size] = True
        
        
        mask = np.real(magnitude_spectrum) > threshold

        fshift[mask] = 0
        
       
        f_ishift = np.fft.ifftshift(fshift)
        magnitude_spectrum_filtered01 = 20 * np.log(np.abs(fshift))
        
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        
        #threshold == None
#        name = '{} frame = {} size = {}'.format(appendix, frame, size)
        name = '{} frame = {} threshold = {}'.format(appendix, frame, threshold)
        
#        plt.suptitle(name)
#        plt.subplot(111),plt.imshow(img_back[:,300:750], cmap = 'gray', vmin=-0.01, vmax=0.01)
##        plt.title('Output image'), plt.xticks([]), plt.yticks([])
##        plt.savefig(folder + 'export_fouriere/output ' + name + '.png', dpi=600)
#        plt.show()
#        
#        plt.suptitle(name)
#
#        plt.subplot(111),plt.imshow(img[:,300:750], cmap = 'gray', vmin=-0.01, vmax=0.01)
##        plt.title('Output image'), plt.xticks([]), plt.yticks([])
##        plt.savefig(folder + 'export_fouriere/input ' + name + '.png', dpi=600)
#        plt.show()
        
        #300:750, :
        
        plt.suptitle(name)
        
        plt.subplot(221),plt.imshow(img, cmap = 'gray', vmin=-0.01, vmax=0.01)
        plt.title('Input image'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray', vmin=-50, vmax=50)
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(223),plt.imshow(mask, cmap = 'gray')
        plt.title('Chosen frequencies'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(224),plt.imshow(img_back, cmap = 'gray', vmin=-0.01, vmax=0.01)
        plt.title('Output image'), plt.xticks([]), plt.yticks([])
        plt.show()
        
        
#        plt.savefig(folder + 'export_fouriere/' + name + '.png', dpi=600)
        
print('{:.2f} s'.format(t.time()-time_start))
