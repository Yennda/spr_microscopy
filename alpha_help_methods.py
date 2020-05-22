import math as m
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from global_var import *


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
    if x >= (x0 - w / 2) and x < (x0 + w / 2):
        return - m.fabs(x - x0) * h / w * 2 + h
    else:
        return 0

def correlation_temporal(
        data, 
        k_diff, 
        step = - 0.003,
        threshold = 15, 
        show = False
        ):
    """
    Temporal correlation of data signal with the trigonal function of defined width (k_diff) and height (step). 
    The size of the step is defined according to the positive binding event.
    
    Parameters:
        k_diff (int): number of integrated frames in sequential referencing. Halfwidth of trigonal function
        step (float): height of the trigonal function. 
        
    Returns:
        frames of supposed binding and unbinding events
        
    """
    correlation = [0] * k_diff
    tri = [
            func_tri(
                    x, 
                    x0 = k_diff, 
                    h = step, 
                    w = k_diff * 2
                    ) 
            for x in range(0, 2 * k_diff)
            ]
    
    for i in range(k_diff, len(data) - k_diff):

        correlation.append(
                np.correlate(
                        data[i - k_diff: i + k_diff], 
                        tri
                        )[0] * 1e5
                )
                
        if np.abs(correlation).max() > threshold:
            peaks_binding, _ = find_peaks(
                    correlation, 
                    height = threshold
                    )
            peaks_unbinding, _ = find_peaks(
                    [- c for c in correlation], 
                    height = threshold
                    )
        else:
            peaks_binding = []
            peaks_unbinding = []
            
    if show:             
        fig, axes = plt.subplots()
        axes.plot(
                data*1000, 
                ls = '-', 
                color = black, 
                label='signal'
                )
#        axes.set_title('info')
        axes.set_xlabel('Frame')
        axes.set_ylabel('Intensity [a. u.]')
        
        axes.plot(
                tri, 
                ls = '-', 
                color = yellow, 
                label='tri. f-n'
                )
        
        axes_corr = axes.twinx()
        axes_corr.set_ylabel('Correlation [a. u.]')
        axes_corr.plot(
                correlation, 
                ls = '-', 
                color = gray, 
                label='corr.'
                )
#        axes_corr.scatter(
#                peaks_binding,
#                [correlation[p] for p in peaks_binding], 
#                label = 'bind.', 
#                color  = green
#                )
#        axes_corr.scatter(
#                peaks_unbinding, 
#                [correlation[p] for p in peaks_unbinding], 
#                label = 'unbind.', 
#                color = red
#                )
        lgd = fig.legend(loc=4)
#        fig.legend(loc = 3)
    

    return {
            'bind' : (peaks_binding, data[peaks_binding]), 
            'unbind' : (peaks_unbinding, data[peaks_unbinding])
            }
