import os

import math as m
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import data, io, filters, util
import matplotlib.patches as mpatches
import tools

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
            
    # wrong order, no tin agreement with the table in the picture
    return [len(out)*SCALE, np.average(np.abs(out))/std]

def profiles(raw, coor):
    list_i=[c[1] for c in coor]
    list_j=[c[0] for c in coor]
    
    
    
    
#    std=(np.std(raw[0:15, :])+np.std(raw[-15:-1, :])+np.std(raw[15:-15, 0:15])+np.std(raw[15:-15, -15:-1]))/4
    std=(np.std(raw[min(list_i), :])+np.std(raw[max(list_i):-1, :])+np.std(raw[min(list_i):max(list_i), 0:min(list_j)])+np.std(raw[min(list_i):max(list_i), max(list_j):-1]))/4
    
    #NP image intensity
    avg=np.average
    raw_a=abs(raw)
    avg_abs=(avg(raw_a[min(list_i), :])+avg(raw_a[max(list_i):-1, :])+avg(raw_a[min(list_i):max(list_i), 0:min(list_j)])+avg(raw_a[min(list_i):max(list_i), max(list_j):-1]))/4
    np_avg_abs=avg(raw_a[min(list_i):max(list_i), min(list_j):max(list_j)])
    
    intensity=np_avg_abs-avg_abs
    


    sizes=[
            (coor[1][0]-coor[0][0])*SCALE,
            (coor[3][1]-coor[2][1])*SCALE,
            np.average(raw[coor[0][1], coor[0][0]:coor[1][0]])/std,
            np.average(raw[coor[2][1]:coor[3][1], coor[2][0]])/std
            ]
    

    return sizes + [std, intensity]
    
def np_analysis(raw, folder='images', file='image_np'):
    
    def mouse_click(event):
        
        x=int((event.xdata+0.5)//1)
        y=int((event.ydata+0.5)//1)
        print(x, y)
        
        fig = event.canvas.figure
        ax = fig.axes[0]
        ax.index+=1
        ax.coor.append([x, y])
        p=mpatches.Rectangle((x-0.5, y-0.5), 1, 1, color='#000000')
        ax.add_patch(p)
        
        if ax.index==4:
            sizes=profiles(raw, ax.coor)
            info='x ={:.01f}$\mu m$\ny ={:.01f}$\mu m$\nCx={:.01f} \nCy={:.04f} \nstd={:.04f}, \nint={:.04f}'.format(*sizes)
            ax.text(0, 14, info, fontsize=10, bbox={'facecolor':'white', 'alpha':0.5, 'pad':1})
            ax.index=0
            
            tools.new_export(folder, '/export_np')
            name='{}/export_np/{}'.format(folder, file)
            
            i=1
            while os.path.isfile(name+'_{:02d}.png'.format(i)):
                i+=1
            name+='_{:02d}'.format(i)
            
            
            
            figgraph = plt.figure()
            axes = figgraph.add_axes([0.1, 0.1, 0.8, 0.8])
            axes.grid(True)
            axes.set_title('Profiles of NP')
            axes.set_xlabel('x')
            axes.set_ylabel('inensity change')
            
            data_x=raw[ax.coor[0][1], 0:50]
            data_y=raw[0:50, ax.coor[2][0]]
            
            std=sizes[4]
            
            axes.plot(np.arange(0, 50, 1), data_x, color='red', label='x')
            axes.plot(np.arange(0, 50, 1), data_y, color='blue', label='y')
            axes.plot(np.arange(0, 50, 1), [std*3]*50, color='gray', label='3*std')
            axes.plot(np.arange(0, 50, 1), [std]*50, color='black', label='std')
            axes.plot(np.arange(0, 50, 1), [-std*3]*50, color='gray')
            axes.plot(np.arange(0, 50, 1), [-std]*50, color='black')
            axes.legend(loc=2) 
            fig.savefig(name+'.png' , dpi=300)
            figgraph.savefig(name+'_graph.png' , dpi=300)
            
            # x, y, Cx, Cy, std, intensity
            with open(name[:-2]+'info.txt', "a+", encoding="utf-8") as f:
                f.write('{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{}\t{}\n'.format(*sizes))
        fig.canvas.draw()     
        
    
    
    
    fig, ax = plt.subplots()
    img=ax.imshow(raw)
    img.set_cmap('Spectral')
    fig.colorbar(img, ax=ax)
    ax.index=0
    ax.coor=[]
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    print('''
Select the points in the following order:
        [3]
    [1]     [2]
        [4]
              ''')
    
    
    
    
    
plt.close("all")

path='particle_01.tif'
raw=io.imread(path)

np_analysis(raw)

