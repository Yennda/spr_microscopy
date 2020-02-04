import cv2
import numpy as np
from matplotlib import pyplot as plt
from video_processing import Video
import copy
import time

# https://lmfit.github.io/lmfit-py/model.html
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

plt.close('all')

# folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'
main_folder = 'C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder = main_folder + '19_08_16_K4/'
folder = main_folder + '20_01_24_third/'

#[200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]

for jj in [1, 2, 5, 10, 20, 30, 40, 50]:

    file = 'raw_01'
    frame = 900
    treshhold = jj
    
    
    # folder=main_folder+'19_07_18_C5/'
    # file='neref_02_1'
    file = 'norm_05_1'
    file = 'raw_01_1'
    
    video = Video(folder, file)
    video.loadData()
    video._video['raw'] = video._video['raw'][:,:,:frame+1]
    video.make_int()
    
    # img = cv2.imread('AR_images/stinkbug.png',0)
    img = video._video['int'][:, :, frame]
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    #
    # middle=int(video._video.shape[2]/2)
    # vid=np.zeros(video._video.shape)
    # for i in range(video._video.shape[0]):
    #    print('done {}/{}'.format(i, video._video.shape[0]))
    #    for j in range(video._video.shape[1]):
    #        signal=video._video[i, j,:]
    #        fspec=np.fft.fft(signal)
    #        
    #        fspec[middle-10:middle+10]=0
    #        print(fspec)
    #
    #
    #        
    #        vid[i, j,:] = np.fft.ifft(fspec)
    #        
    # plt.imshow(vid[:,:,-1])
    # video._video=vid
    #
    #signal = video._video['raw'][31, 1102, :]
    ##signal = video.video[1102, 31, :]
    #fspec = np.fft.fft(signal)
    ## magnitude_spectrum_time = 20*np.log(np.abs(fshift))
    #
    #ffilt = copy.deepcopy(fspec)
    #ffilt2 = copy.deepcopy(fspec)
    #ffilt[5:35] = 0
    #ffilt2[:15] = 0
    ## ffilt2[10:30]=0.
    #ffilt2[25:] = 0
    ## ffilt2[19:21]=fspec[19:21]
    #
    #
    #signal_back = np.fft.ifft(ffilt)
    #signal_back2 = np.fft.ifft(ffilt2)
    #
    #fix, axes = plt.subplots()
    #axes.plot(fspec, 'b-', label='fouriere-spectrum')
    ##axes.plot(signal_back, 'r-', label='fouriere-filtered')
    #axes.legend(loc=3)
    #
    #fix2, axes2 = plt.subplots()
    #axes2.plot(signal, 'b-', label='signal')
    #axes2.plot(signal_back, 'r-', label='fouriere-filtered 1st')
    #axes2.plot(signal_back2, 'g-', label='fouriere-filtered 2nd')
    #axes2.legend(loc=3)
    
    
    
    # f_line=np.asarray(f).reshape(-1)
    # plt.clf()
    # plt.hist(np.abs(f_line), 1000)
    # plt.xscale('log')
    # plt.show()
    
    
    mask = np.real(magnitude_spectrum) > treshhold
    
    #only strips, efficient
    #fshift[:, 490:550] = 0
    #fshift[:, 1130:1190] = 0
    fshift[mask] = 0
    
    # rows, cols = img.shape
    # crow,ccol = int(rows/2) , int(cols/2)
    # fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    
    #fshift[111:136, 139:309] = 0
    #fshift[133:163, 1318:1522] = 0
    
    
    f_ishift = np.fft.ifftshift(fshift)
    magnitude_spectrum_filtered01 = 20 * np.log(np.abs(fshift))
    
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    
    #treshhold == None
    name = 'frame = {} treshhold = {}'.format(frame, treshhold)
    
    
    plt.suptitle(name)
    plt.subplot(111),plt.imshow(img_back[:,300:750], cmap = 'gray', vmin=-0.01, vmax=0.01)
    plt.title('Output image'), plt.xticks([]), plt.yticks([])
    plt.savefig(folder + 'export_fouriere/output ' + name + '.png', dpi=600)
    plt.show()
    
    
    plt.suptitle(name)
    
    plt.subplot(221),plt.imshow(img[:,300:750], cmap = 'gray', vmin=-0.01, vmax=0.01)
    plt.title('Input image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray', vmin=-50, vmax=50)
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(223),plt.imshow(mask, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(224),plt.imshow(img_back[:,300:750], cmap = 'gray', vmin=-0.01, vmax=0.01)
    plt.title('Output image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    plt.savefig(folder + 'export_fouriere/' + name + '.png', dpi=600)
