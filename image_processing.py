import os
import math as m
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
import tools
from scipy.optimize import curve_fit
from tools import COLORS
from scipy.signal import find_peaks
from tools import COLORS

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
    
def correlation_temporal(data, k_diff, step, show=False):
    """
    Temporal correlation of data signal with the trigonal function of defined width (k_diff) and height (step). 
    The size of the step is defined according to the positive binding event.
    
    Parameters:
        k_diff (int): number of integrated frames in sequential referencing. Halfwidth of trigonal function
        step (float): height of the trigonal function. 
        
    Returns:
        no return
        
    """
    correlation = [0]*k_diff
    for i in range(k_diff, len(data)-k_diff):
        
        tri = [func_tri(x, x0 = i, h = step, w = k_diff*2) for x in range(i-k_diff, i+k_diff)]
        correlation.append(np. correlate(data[i-k_diff:i+k_diff], tri)[0]*1e5)
        
        peaks_binding, _ = find_peaks(correlation, height=15)
        peaks_unbinding, _ = find_peaks([-c for c in correlation], height=15)
        
    if show:             
        fig, axes = plt.subplots()
        axes.plot(data, ls = '-', color = COLORS[1], label='data')
    #        axes.set_xlim(0, 5)
    #        axes.set_ylim(-0.0035, 0.0005)
        axes.plot(tri, ls = '-', color = COLORS[0], label='triangular function')
        axes_corr = axes.twinx()
        axes_corr.plot(correlation, ls = '-', color = COLORS[2], label='correlation')
        axes_corr.scatter(peaks_binding, [correlation[p] for p in peaks_binding], label='binding', color=COLORS[1])
        axes_corr.scatter(peaks_unbinding, [correlation[p] for p in peaks_unbinding], label='unbinding', color=COLORS[2])
        fig.legend(loc=3)
        

    
#    print(peaks_binding)
#    print(peaks_unbinding)
    return (peaks_binding, peaks_unbinding)   