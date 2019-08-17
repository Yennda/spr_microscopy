from video_processing import Video
import numpy as np
import math as m
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

def h(x): return 0.5 * (np.sign(x) + 1)

def step(x, a, b): return b*0.5 * (np.sign(x-a) + 1)

def chute(x, a0, a1, b0, b1):
    return b0*h(a0-x)+b1*h(x-a1)+((b1-b0)/(a1-a0)*(x-a0)+b0)*h(x-a0)*h(a1-x)

def t2i(boo):
    if boo:
        return 1
    else:
        return 0

def is_np(inten, treshold=1e-07, show=False):

    xdata=np.arange(len(inten))
    popt, pcov = curve_fit(step, xdata, inten, p0=[10,-1e-03], epsfcn=0.01)
    squares=sum([(step(i, *popt)-inten[i])**2 for i in xdata])
    if show:
        print('a, b: {}'.format(popt))
        #    print(pcov)
        squares=sum([(step(i, *popt)-inten[i])**2 for i in xdata])
        print('squares: {}'.format(squares))
        print('variance: {}'.format(np.var(inten)))
        plt.plot(inten, 'b-', label='data')  
        plt.plot(xdata, step(xdata, *popt), 'r-')
#    return popt[1]-popt[0]<4 # and m.fabs(popt[3]-popt[2])>1e-04
    return np.var(inten)>treshold and m.fabs(popt[1])>1e-04 and squares>1e-06



plt.close("all")
folder='C:/Users/jabuk/Documents/jaderka/ufe/data/'
#vyska, sirka, cas
file='norm_05_1'


video=Video(folder, file)
video.loadData()

data=video._video[40:160, 1000:1200, :]
del video
#plt.imshow(data[:,:,22])

mask=np.zeros(data[:,:,0].shape)

for i in range(data.shape[0]):
    print('done {}/{}'.format(i, data.shape[0]))
    for j in range(data.shape[1]):
        try:
            mask[i,j]=t2i(is_np(data[i,j,:]))
        except:
            mask[i,j]=0
            print('no fit')
print('done')

plt.imshow(mask)
#print(is_np(data[45,66,:], show=True))
#print(is_np(data[68,135,:], show=True))
#print(is_np(data[50,135,:], show=True))
#plt.show()











