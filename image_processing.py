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
    
#k_diffo = 10
#stepo = -0.0055
#tri = [func_tri(x, x0 = k_diffo, h = stepo, w = k_diffo*2) for x in range(0, 2*k_diffo)]

def correlation_temporal(data, k_diff, step=-0.003, threshold = 15, show = False):
    """
    Temporal correlation of data signal with the trigonal function of defined width (k_diff) and height (step). 
    The size of the step is defined according to the positive binding event.
    
    Parameters:
        k_diff (int): number of integrated frames in sequential referencing. Halfwidth of trigonal function
        step (float): height of the trigonal function. 
        
    Returns:
        frames of supposed binding and unbinding events
        
    """
    correlation = [0]*k_diff
    tri = [func_tri(x, x0 = k_diff, h = step, w = k_diff*2) for x in range(0, 2*k_diff)]
    
    for i in range(k_diff, len(data)-k_diff):

        correlation.append(np.correlate(data[i-k_diff:i+k_diff], tri)[0]*1e5)
        if np.abs(correlation).max() > threshold:
            peaks_binding, _ = find_peaks(correlation, height=threshold)
            peaks_unbinding, _ = find_peaks([-c for c in correlation], height=threshold)
        else:
            peaks_binding = []
            peaks_unbinding = []
    if show:             
        fig, axes = plt.subplots()
        axes.plot(data, ls = '-', color = COLORS[3], label='signal')
    #        axes.set_xlim(0, 5)
    #        axes.set_ylim(-0.0035, 0.0005)
        axes.plot(tri, ls = '-', color = COLORS[0], label='tri. f-n')
        axes_corr = axes.twinx()
        axes_corr.plot(correlation, ls = '-', color = COLORS[1], label='corr.')
        axes_corr.scatter(peaks_binding, [correlation[p] for p in peaks_binding], label='bind.', color=COLORS[4])
        axes_corr.scatter(peaks_unbinding, [correlation[p] for p in peaks_unbinding], label='unbind.', color=COLORS[2])
        fig.legend(loc=3)
    

    
#    print(peaks_binding)
#    print(peaks_unbinding)
    return (peaks_binding, peaks_unbinding, data[peaks_binding], data[peaks_unbinding])   