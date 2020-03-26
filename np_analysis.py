import os
import math as m
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector

from PIL import Image
import tools
from scipy.optimize import curve_fit
from tools import COLORS
from scipy.signal import argrelextrema

SCALE = 2.93  # mu/px
SHAPE = 50  # dimension of the image in px

def measure_new(raw, mask_np, sizes):
    mask_background = mask_np == False
    
    std = np.std(raw[mask_background])
    int_np = sum(abs(raw[mask_np]))
    int_bg_px = sum(abs(raw[mask_background])) / len(raw[mask_background])
    
    int_np_norm = int_np - int_bg_px * len(raw[mask_np])
    int_np_norm_px = int_np_norm / len(raw[mask_np])

    contrast = int_np_norm_px / int_bg_px
    sizes = [s*SCALE for s in sizes]
     
    out = sizes + [contrast, int_np_norm*1e3, int_np_norm_px*1e3, int_bg_px*1e3, std*1e3]
    
    for i in range(len(out)):
        if m.isnan(out[i]):
            out[i] = 0
#    fig, ax = plt.subplots()
#    ax.imshow(raw)
#    ax.imshow(mask_np, alpha = 0.5)      
    
    return out

def visualize_and_save(raw, measures, name):
#    fig, ax = plt.subplots()
#    img = ax.imshow(raw)
#    img.set_cmap('Greys')
#    
#
#    info = 'x ={:.01f}$\mu m$\ny ={:.01f}$\mu m$\nC={:.01f}\n$I_n$ np={:.04f}, \n$I_n$/px={:.04f}\n$I_b$/px={:.04f}\nstd={:.04f}'.format(*measures)
#
#    
#    ax.text(0, 12, info, fontsize=10, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
#    ax.index = 0

    with open(name, "a+", encoding="utf-8") as f:
        f.write('{:.02f}\t{:.02f}\t{}\t{}\t{}\t{}\t{}\n'.format(*measures))
        
def select_idea(raw):
    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['X', 'x'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    fig, ax = plt.subplots()
    img = ax.imshow(
            raw,
            cmap='gray'
            )
    
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
    
    plt.connect('key_press_event', toggle_selector)
    

def stats(raw, p=False):
    avg = np.average(raw)
    avg_abs = np.average(abs(raw))
    std = (np.std(raw[0:15, :]) + np.std(raw[-15:-1, :]) + np.std(raw[15:-15, 0:15]) + np.std(raw[15:-15, -15:-1])) / 4
    std_abs = np.std(abs(raw))
    mn = np.min(raw)
    mx = np.max(raw)

    if p:
        print('avg_abs = {}'.format(avg_abs))
        print('avg = {}'.format(avg))
        print('std = {}'.format(std))
        print('std_abs = {}'.format(std_abs))
        print('min = {}'.format(mn))
        print('max = {}'.format(mx))
    return avg, std, mn, mx


def intensity(raw):
    raw_a = abs(raw)
    avg = np.average
    avg_abs = (avg(raw_a[0:15, :]) + avg(raw_a[-15:-1, :]) + avg(raw_a[15:-15, 0:15]) + avg(raw_a[15:-15, -15:-1])) / 4
    np_avg_abs = avg(raw_a[15:-15, 15:-15])
    return np_avg_abs - avg_abs


def size(data, std):
    std = abs(std)
    out = []
    for i in range(len(data)):
        if abs(data[i]) > 3 * std and (abs(data[i + 1]) > 3 * std or abs(data[i - 1]) > 3 * std):
            out.append(data[i])
            print(i)
        elif abs(data[i]) > 5 * std:
            out.append(data[i])
            print(i)

    # wrong order, no thin agreement with the table in the picture
    return [len(out) * SCALE, np.average(np.abs(out)) / std]


def measure(raw, coor):
    list_i = [c[0] for c in coor]
    list_j = [c[1] for c in coor]

    indices_i = [i for i in range(0, list_i[0] + 1)]
    indices_i += [i for i in range(list_i[1], 50)]
    indices_j = [i for i in range(0, list_j[2] + 1)]
    indices_j += [i for i in range(list_j[3], 50)]

    #    std=(np.std(raw[0:15, :])+np.std(raw[-15:-1, :])+np.std(raw[15:-15, 0:15])+np.std(raw[15:-15, -15:-1]))/4
    mask_background = np.array([[(i in indices_i) or (j in indices_j) for i in range(SHAPE)] for j in range(SHAPE)])
    mask_np = np.array([[not ((i in indices_i) or (j in indices_j)) for i in range(SHAPE)] for j in range(SHAPE)])

    std = np.std(raw[mask_background])
    int_np = sum(abs(raw[mask_np]))
    int_background = sum(abs(raw[mask_background])) / len(raw[mask_background]) * len(raw[mask_np])
    rel_background = sum(abs(raw[mask_background])) / len(raw[mask_background])
    intensity = int_np - int_background
    max_int=np.max(np.absolute(raw[mask_np]))
    contrast = int_np / len(raw[mask_np]) / std

    #    fig, ax = plt.subplots()
    #    raww=raw
    #    raw[list_j[2]+1:list_j[3]+1, list_i[2]]=0.01
    #    img=ax.imshow(raww)

    sizes = [
        (list_i[1] - list_i[0] - 1) * SCALE,
        (list_j[3] - list_j[2] - 1) * SCALE,
        np.average(abs(raw[list_j[0], list_i[0] + 1:list_i[1]])) / std,
        np.average(abs(raw[list_j[2] + 1:list_j[3], list_i[2]])) / std
    ]
    print(contrast)
    return sizes + [contrast, std, intensity, max_int, rel_background]

def np_analysis(raw, folder='images', file='image_np'):
    def mouse_click(event):

        x = int((event.xdata + 0.5) // 1)
        y = int((event.ydata + 0.5) // 1)
        #        print(x, y)

        fig = event.canvas.figure
        ax = fig.axes[0]
        ax.index += 1
        ax.coor.append([x, y])
        p = mpatches.Rectangle((x - 0.5, y - 0.5), 1, 1, color='#000000')
        ax.add_patch(p)

        if ax.index == 4:
            measures = measure(raw, ax.coor)

            info = 'x ={:.01f}$\mu m$\ny ={:.01f}$\mu m$\nCx={:.01f} \nCy={:.01f}\nC={:.01f}\nstd={:.04f}, \nint={:.04f}\nmaxint={:.04f}\nrelbg={:.04f}'.format(
                *measures)
            ax.text(0, 12, info, fontsize=10, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
            ax.index = 0

            tools.new_export(folder, '/export_np')

            name = '{}/export_np/{}'.format(folder, file)
            i = 1
            while os.path.isfile(name + '_{:02d}.png'.format(i)):
                i += 1
            name += '_{:02d}'.format(i)

            figgraph = plt.figure()
            axes = figgraph.add_axes([0.1, 0.1, 0.8, 0.8])
            axes.grid(True)
            axes.set_title('Profiles of NP')
            axes.set_xlabel('x')
            axes.set_ylabel('inensity change')

            data_x = raw[ax.coor[0][1], 0:50]
            data_y = raw[0:50, ax.coor[2][0]]

            std = measures[5]

            axes.plot(np.arange(0, 50, 1), data_x, color='red', label='x')
            axes.plot(np.arange(0, 50, 1), data_y, color='blue', label='y')
            axes.plot(np.arange(0, 50, 1), [std * 3] * 50, color='gray', label='3std')
            axes.plot(np.arange(0, 50, 1), [std] * 50, color='black', label='std')
            axes.plot(np.arange(0, 50, 1), [-std * 3] * 50, color='gray')
            axes.plot(np.arange(0, 50, 1), [-std] * 50, color='black')
            axes.legend(loc=2)
            fig.savefig(name + '.png', dpi=300)
            figgraph.savefig(name + '_graph.png', dpi=300)

            pilimage = Image.fromarray(img.get_array())
            pilimage.save(name + '.tiff')
            print('File SAVED @{}'.format(name))

            # x, y, Cx, Cy, std, intensity
            with open(name[:-2] + 'info.txt', "a+", encoding="utf-8") as f:
                f.write('{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{}\t{}\t{}\t{}\t{}\n'.format(*measures))

            #            plt.close(2)
            plt.close(3)
        fig.canvas.draw()

    fig, ax = plt.subplots()
    img = ax.imshow(raw)
    img.set_cmap('Spectral')
    fig.colorbar(img, ax=ax)
    ax.index = 0
    ax.coor = []
    fig.canvas.mpl_connect('button_press_event', mouse_click)
    print('''
Select the points in the following order:
        [3]
    [1]     [2]
        [4]
              ''')


def h(x): return 0.5 * (np.sign(x) + 1)


def step(x, a, b0, b1): return (b1 - b0) * (np.sign(x - a) + 1) + b0


def linear(x, a, b):
    return a * x + b


def find_step(data):
    return np.argmax([m.fabs(data[i] - data[i + 2]) for i in range(len(data) - 2)])

def func_tri(x, x0, h, w):
    """
   Triangular function
    
    Parameters:
        x0 (float): position of the centre of the triangle
        h (float): height of the triangle
        w (float): width of the triangle
        
    Returns:
        y value of a triangular function, based on the parameters x0, h and w
        
    """
    if x >= (x0 - w/2) and x < (x0 + w/2):
        return -m.fabs(x-x0)*h/w*2+h
    else:
        return 0
    

    
    
def is_np(data, inten_a=1e-04, inten_b=5e-4, show=False):
#    xdata = np.arange(len(data))
    correlation = [0]*10
    for i in range(10, len(data)-10):
        tri = [func_tri(x, x0 = i, h = -0.0055, w = 20) for x in range(i-10, i+10)]
#        print(np. correlate(data, tri))
        correlation.append(np. correlate(data[i-10:i+10], tri)[0]*1e5)
    
#    y = [func_tri(x, x0 = 20, h = -0.01, w = 20) for x in range(len(data))]

    if show:

        fig, axes = plt.subplots()
        axes.plot(data, ls = '-', color = COLORS[1], label='data')
#        axes.set_xlim(0, 5)
#        axes.set_ylim(-0.0035, 0.0005)
        axes.plot(tri, ls = '-', color = COLORS[0], label='triangular function')
        axes_corr = axes.twinx()
        axes_corr.plot(correlation, ls = '-', color = COLORS[2], label='correlation')
        
        fig.legend(loc=3)

