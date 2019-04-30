import os

import math as m
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import data, io, filters, util

SCALE=2.93 # mu/px

def stats(raw, p=False):
    avg=np.average(raw)
    avg_abs=np.average(abs(raw))
    std=(np.std(raw[0:15, :])+np.std(raw[-15:-1, :])+np.std(raw[15:-15, 0:15])+np.std(raw[15:-15, -15:-1]))/4
    std_abs=np.std(abs(raw))
    mn=np.min(raw)
    mx=np.max(raw)
    
    if p:
        print('avg_abs = {}'.format(avg_abs))
        print('avg = {}'.format(avg))
        print('std = {}'.format(std))
        print('std_abs = {}'.format(std_abs))
        print('min = {}'.format(mn))
        print('max = {}'.format(mx))
    return avg, std, mn, mx

def intensity(raw):
    raw_a=abs(raw)
    avg=np.average
    avg_abs=(avg(raw_a[0:15, :])+avg(raw_a[-15:-1, :])+avg(raw_a[15:-15, 0:15])+avg(raw_a[15:-15, -15:-1]))/4
    np_avg_abs=avg(raw_a[15:-15, 15:-15])
    return np_avg_abs-avg_abs
    
def size(data, std):
    std=abs(std)
    out=[]
    for i in range(len(data)):
        if abs(data[i])>3*std and (abs(data[i+1])>3*std or abs(data[i-1])>3*std):
            out.append(data[i])
            print(i)
        elif abs(data[i])>5*std:
            out.append(data[i])
            print(i)
            
    
    return [len(out)*SCALE, np.average(np.abs(out))/std]

def np_analysis(raw):
    std=(np.std(raw[0:15, :])+np.std(raw[-15:-1, :])+np.std(raw[15:-15, 0:15])+np.std(raw[15:-15, -15:-1]))/4
    data_x=raw[25, 0:50]
    data_y=raw[0:50, 25]
    #data_y=[np.average(raw[i, 23:27]) for i in range(50)]
    
    sizes=size(data_x, std)+size(data_y, std)
    
    print(sizes)
    print(intensity(raw))
    
    
    fig, ax = plt.subplots()
    img=ax.imshow(raw, interpolation='nearest')
    img.set_cmap('binary_r')
    
    print(sizes)
    info='x ={:.01f}$\mu m$\ny ={:.01f}$\mu m$\nCx={:.01f} \nCy={:.01f}'.format(*sizes)
    
    ax.text(0, 10, info, fontsize=10, bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
    fig.colorbar(img, ax=ax)
    
    
    
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.grid(True)
    axes.set_title('title')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    
    axes.plot(np.arange(0, 50, 1), data_x)
    axes.plot(np.arange(0, 50, 1), data_y)
    axes.plot(np.arange(0, 50, 1), [std*3]*50, color='blue')
    axes.plot(np.arange(0, 50, 1), [-std*3]*50, color='blue')
    axes.plot(np.arange(0, 50, 1), [std]*50, color='red')
    axes.plot(np.arange(0, 50, 1), [-std]*50, color='red')
    
    
    
#plt.close("all")
#
#path='particle_02.tif'
#raw=io.imread(path)
#
#np_analysis(raw)



#img=plt.imshow(np.abs(raw), interpolation='nearest')
#img.set_cmap('binary_r')
#plt.colorbar()
##img.set_clim([-0.01, 0.01])
#
#std=(np.std(raw[0:15, :])+np.std(raw[-15:-1, :])+np.std(raw[15:-15, 0:15])+np.std(raw[15:-15, -15:-1]))/4
#data_x=raw[25, 0:50]
#data_y=raw[0:50, 25]
##data_y=[np.average(raw[i, 23:27]) for i in range(50)]
#
#sizes=[size(data_x, std), size(data_y, std)]
#
#print(sizes)
#print(intensity(raw))
#
#fig = plt.figure()
#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#axes.grid(True)
#axes.set_title('title')
#axes.set_xlabel('x')
#axes.set_ylabel('y')
#
#axes.plot(np.arange(0, 50, 1), data_x)
#axes.plot(np.arange(0, 50, 1), data_y)
#axes.plot(np.arange(0, 50, 1), [std*3]*50, color='blue')
#axes.plot(np.arange(0, 50, 1), [-std*3]*50, color='blue')
#axes.plot(np.arange(0, 50, 1), [std]*50, color='red')
#axes.plot(np.arange(0, 50, 1), [-std]*50, color='red')
