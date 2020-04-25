import math as m
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

SCALE = 2.93  # mu/px
SHAPE = 50  # dimension of the image in px

def characterize(raw, mask_np, sizes):
    mask_background = mask_np == False
    
    std = np.std(raw[mask_background])
    int_np = sum(abs(raw[mask_np]))
    int_bg_px = sum(abs(raw[mask_background])) / len(raw[mask_background])
    
    int_np_norm = int_np - int_bg_px * len(raw[mask_np])
    int_np_norm_px = int_np_norm / len(raw[mask_np])

    contrast = int_np_norm_px / int_bg_px
    sizes = [s*SCALE for s in sizes]
     
    out = sizes + [
            contrast, 
            int_np_norm*1e3, 
            int_np_norm_px*1e3, 
            int_bg_px*1e3, 
            std*1e3
            ]
    
    for i in range(len(out)):
        
        if m.isnan(out[i]):
            out[i] = 0
            
#    fig, ax = plt.subplots()
#    ax.imshow(raw, cmap = 'gray')
#    ax.imshow(mask_np, cmap = 'autumn', alpha = 0.8)  
#    fontprops = fm.FontProperties(size=20)
#    scalebar = AnchoredSizeBar(ax.transData,
#                   34, '100 $\mu m$', 'lower right', 
#                   pad=0.1,
#                   color='black',
#                   frameon=False,
#                   size_vertical=1,
#                   fontproperties=fontprops)
#
#    ax.add_artist(scalebar)    
    return out

def save(raw, measures, name):
    with open(name, "a+", encoding="utf-8") as f:
        f.write('{:.02f}\t{:.02f}\t{}\t{}\t{}\t{}\t{}\n'.format(*measures))
        



