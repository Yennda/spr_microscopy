import os
import math as m
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io

import tools

SCALE=2.93 # mu/px
SHAPE=50    #dimension of the image in px

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

def measure(raw, coor):
    list_i=[c[0] for c in coor]
    list_j=[c[1] for c in coor]
    
    indices_i=[i for i in range(0, list_i[0]+1)]
    indices_i+=[i for i in range(list_i[1], 50)]
    indices_j=[i for i in range(0, list_j[2]+1)]
    indices_j+=[i for i in range(list_j[3], 50)]

#    std=(np.std(raw[0:15, :])+np.std(raw[-15:-1, :])+np.std(raw[15:-15, 0:15])+np.std(raw[15:-15, -15:-1]))/4
    mask_background=np.array([[(i in indices_i) or (j in indices_j) for i in range(SHAPE)] for j in range(SHAPE)])
    mask_np=np.array([[not ((i in indices_i) or (j in indices_j)) for i in range(SHAPE)] for j in range(SHAPE)])
    
    std=np.std(raw[mask_background])
    int_np=sum(abs(raw[mask_np]))
    int_background=sum(abs(raw[mask_background]))/len(raw[mask_background])*len(raw[mask_np])
    rel_background=sum(abs(raw[mask_background]))/len(raw[mask_background])
    intensity=int_np-int_background
    contrast=int_np/len(raw[mask_np])/std
    
#    fig, ax = plt.subplots()
#    raww=raw
#    raw[list_j[2]+1:list_j[3]+1, list_i[2]]=0.01
#    img=ax.imshow(raww)
      
    sizes=[
            (list_i[1]-list_i[0]-1)*SCALE,
            (list_j[3]-list_j[2]-1)*SCALE,
            np.average(abs(raw[list_j[0], list_i[0]+1:list_i[1]]))/std,
            np.average(abs(raw[list_j[2]+1:list_j[3], list_i[2]]))/std
            ]
    print(contrast)
    return sizes + [contrast, std, intensity, rel_background]
    
def np_analysis(raw, folder='images', file='image_np'):
    
    def mouse_click(event):
        
        x=int((event.xdata+0.5)//1)
        y=int((event.ydata+0.5)//1)
#        print(x, y)
        
        fig = event.canvas.figure
        ax = fig.axes[0]
        ax.index+=1
        ax.coor.append([x, y])
        p=mpatches.Rectangle((x-0.5, y-0.5), 1, 1, color='#000000')
        ax.add_patch(p)
        
        if ax.index==4:
            measures=measure(raw, ax.coor)
            
            info='x ={:.01f}$\mu m$\ny ={:.01f}$\mu m$\nCx={:.01f} \nCy={:.01f}\nC={:.01f}\nstd={:.04f}, \nint={:.04f}\nrelbg={:.04f}'.format(*measures)
            ax.text(0, 12, info, fontsize=10, bbox={'facecolor':'white', 'alpha':1, 'pad':1})
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
            
            std=measures[5]
            
            axes.plot(np.arange(0, 50, 1), data_x, color='red', label='x')
            axes.plot(np.arange(0, 50, 1), data_y, color='blue', label='y')
            axes.plot(np.arange(0, 50, 1), [std*3]*50, color='gray', label='3std')
            axes.plot(np.arange(0, 50, 1), [std]*50, color='black', label='std')
            axes.plot(np.arange(0, 50, 1), [-std*3]*50, color='gray')
            axes.plot(np.arange(0, 50, 1), [-std]*50, color='black')
            axes.legend(loc=2) 
            fig.savefig(name+'.png' , dpi=300)
            figgraph.savefig(name+'_graph.png' , dpi=300)
            
            # x, y, Cx, Cy, std, intensity
            with open(name[:-2]+'info.txt', "a+", encoding="utf-8") as f:
                f.write('{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{}\t{}\t{}\t{}\n'.format(*measures))
                
            plt.close(2)
            plt.close(3)
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
    
    
    
#    
#    
#plt.close("all")
#
#path='particle_01.tif'
#raw=io.imread(path)
#
#np_analysis(raw)

