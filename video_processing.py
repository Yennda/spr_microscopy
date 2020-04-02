import os
import warnings
import numpy as np
import math as m
import scipy as sc
import cv2
import copy
import time as tt

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from PIL import Image
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit

import characterization
import alpha_help_methods as ahm
import tools as tl
from global_var import *
from nanoparticle import NanoParticle

warnings.filterwarnings('ignore', category=RuntimeWarning)
            
class Video(object):

    def __init__(self, folder, file):
        self.folder = folder
        self.file = file
        self.file_name = folder + file
        self.__video_stats = None
        
        self._video = {
                'raw': None,
                'diff': None,
                'int': None,
                'corr': None
                }
        self.__toggle = False
        self._img_type = 'raw'
        self.rng = [-1, 1]
        self.time_info = None

        self.ref_frame = 0
        self.k_diff = None
        self.k_int = None
        
        #image processing
        self._img_proc_type = None
        self.mask = None        
        self.candidates = None
        self.px_for_image_mask = []

        self._dict_of_patterns = None
        self.np_database = []
        self.frame_np_ids = []
        
        #statistics
        self.stats_std = None
        self.np_count_first_occurance = None
        self.np_count_present = None
        self.np_count_integral = None
        self.validity = None
        self.valid = None
        
        #settings
        self.threshold = 4
        self.dip = -0.003
        self.noise_level = 0.001
        self._idea_frame = None
        self._idea_span_x = None
        self._idea_span_y = None
        self.idea3d = None
        
        
        #show
        self.show_graphic = True
        self.show_pixels = False
        self.show_detected = False
        self.show_detected_all = False
        self.show_stats = False
        self.show_mask = True
        
        self.info = '=' * 60 + '\nINFO:\n'
        
    def __iter__(self):
        self.n = -1
        self.MAX = self.shape[2] - 1
        return self

    def __next__(self):
        if self.n < self.MAX:
            self.n += 1
            return self.video[:, :, self.n]
        else:
            raise StopIteration

    @property
    def video(self):
        if len(self._img_type) == 1:
            return np.swapaxes(
                    np.swapaxes(self._video[self._img_type[0]], 0, 2), 1, 2
                    )
        else:
            self.__toggle = not self.__toggle
            return np.swapaxes(np.swapaxes(
                    self._video[self._img_type[int(not self._toggle)]]
                    , 0, 2), 1, 2)
        
    def video_from(self, ch):
        if type(ch) == int:
            return np.swapaxes(np.swapaxes(
                    self._video[self._img_type[ch]]
                    , 0, 2), 1, 2)
        elif type(ch) == str:
            return np.swapaxes(np.swapaxes(
                    self._video[ch]
                    , 0, 2), 1, 2)
        else:
            raise ValueError('Unrecognized video type')
            
    @property
    def shape(self):
        sh = self._video['raw'].shape
        return (sh[2], sh[0], sh[1])
    
    @property
    def length(self):
        return self._video['raw'].shape[2]
    
    @property
    def _toggle(self):
        if len(self._img_type) == 1:
            return True
        else:
            return self.__toggle
        
    def info_add(self, text):
        self.info += '{}\n'.format(text)
        
    def loadData(self):
        self.__video_stats = self.loadBinVideoStats()
        self._video['raw'] = self.loadBinVideo()

    def loadBinVideoStats(self):
        suffix = '.tsv'
        with open(self.file_name + suffix, mode='r') as fid:
            file_content = fid.readlines()

        self.time_info = tl.frame_times(file_content)
        stats = file_content[1].split()
        video_length = len(file_content) - 1
        video_width = int(stats[1])
        video_hight = int(stats[2])
        video_fps = float(stats[4]) * int(stats[5])
        
        return [video_fps, [video_width, video_hight, video_length]]

    def loadBinVideo(self):
        code_format = np.float64  # type double
        suffix = '.bin'
        with open(self.file_name + suffix, mode='rb') as fid:
            video = np.fromfile(fid, dtype=code_format)
            fid.close()
            
        video = np.reshape(video, (self.__video_stats[1][0],
                                   self.__video_stats[1][1],
                                   self.__video_stats[1][2]), order='F')

        return np.swapaxes(video, 0, 1)

    def process_diff(self, k = 1):
        sh = self._video['raw'].shape
        out = np.zeros(sh)
        out[:, :, :2 * k] = np.zeros((sh[0], sh[1], 2*k))
        
        print('Differential image') 
        
        for i in range(2 * k, sh[-1]):   
            print('\r\t{}/ {}'.format(i + 1, self.length), end = '')
            current = np.sum(
                    self._video['raw'][:, :, i - k + 1: i + 1], 
                    axis = 2
                    ) / k
            previous = np.sum(
                    self._video['raw'][:, :, i - 2 * k + 1: i - k + 1], 
                    axis = 2
                    ) / k
                    
            out[:, :, i] = current - previous
            
        self.k_diff = k
        
        print(' DONE')
        return out
    
    def process_int(self, k = 1):
        sh = self._video['raw'].shape
        out = np.zeros(sh)
        out[:, :, 0] = np.zeros(sh[0: 2])
        reference = np.sum(
                self._video['raw'][:, :, self.ref_frame: self.ref_frame + k], 
                axis = 2
                ) / k
        
        print('Integral image')
        
        for i in range(1, sh[-1]):
            print('\r\t{}/ {}'.format(i + 1, self.length), end = '')
            out[:, :, i] = self._video['raw'][:, :, i] - reference
        self.k_int = k
            
        print(' DONE')
        return out
    
    def process_corr(self):
        if type(self.idea3d) == type(None):
            raise ValueError('The image of the NP pattern not defined.') 
            
        print('Correlation')
        
        out = sc.signal.correlate(
                self._video['diff'], 
                self.idea3d, 
                mode = 'same'
                ) * 1e5
                
        print('\tDONE')
        return out
        
    def process_mask_image(self):
        volume_mask = np.zeros(list(self.shape) + [4])
        k_diff = self.k_diff
#        k_diff = 1
        tri = [ahm.func_tri(i, k_diff, 0.5, k_diff) for i in range(int(k_diff*2))]

#        for pm in self.px_for_image_mask:
#            f, y, x = pm
#            volume_mask[f, y, x, :3] = tl.hex_to_list(blue)
#            volume_mask[f, y, x, 3] = 0.5
#        return volume_mask     
    
        for pm in self.px_for_image_mask:
            f, y, x = pm
            if f+k_diff > self.video.shape[0]:
                end = self.video.shape[0]
            else:
                end = f+k_diff
            volume_mask[f-k_diff:end, y, x, 1] = [1]*(end-f+k_diff)
            volume_mask[f-k_diff:end, y, x, 3] = tri[:end-f+k_diff]
            
#            if f+k_diff > self.video.shape[0]:
#                end = self.video.shape[0]
#            else:
#                end = f+k_diff
#            volume_mask[f-k_diff:end, y, x, 0] = [1]*(end-f+k_diff)
#            volume_mask[f-k_diff:end, y, x, 3] = tri[:end-f+k_diff]
            
        return volume_mask
    
    def process_mask(self):
        volume_mask = np.zeros(self.shape)

        for c in self.candidates:
            volume_mask[c] = 1

        return volume_mask
    
    def process_arbitrary(self, name, k = 1):
        if name == 'diff':
            out = self.process_diff(k)
        elif name == 'int':
            out = self.process_int(k)
        elif name == 'corr':
            out = self.process_corr()
        elif name == 'raw':
            out = self._video['raw']
        else:
            out = None
            
        return out
    
    def process_fn(self, fn):      
        i = 0
        out = []
        print('Statistics')
        
        for v in self.video_from(0):
            out.append(fn(v))
            i += 1
            print('\r\t{}/ {}'.format(i + 1, self.shape[0]), end = '')
            
        print(' DONE')
        return np.array(out)

    def make_frame_stats(self):
        self.stats_std = self.process_fn(np.std)
        self.stats_min = self.process_fn(np.min)
        self.stats_max = self.process_fn(np.max)
        self.show_stats = True
        
    def make_diff(self, k = 1):
        self._video['diff'] = self.process_diff(k)
        self._img_type = ['diff']
        self.rng = [-0.01, 0.01]

    def make_int(self, k = 1):
        self._video['int']= self.process_int(k)
        self._img_type = ['int']
        self.rng = [-0.01, 0.01]
        
    def make_corr(self):
        self._video['corr']= self.process_corr()
        self._img_type = ['corr']
        
    def make_toggle(self, img_type, k):
        """
        Process two types of video data, 
        the first one serves as the source for the statistics
        
        Parameters:
            img_type (list): list of names of image types, 
            only 2 can be specified, e.g. ['diff', 'int']
            
            k (list): list of parameters for the image processing
            
        Returns:
            no return
            
        """
        if len(img_type) > 2:
            raise ValueError(
                    'Only 2 image types can be processed. You added {}.'
                    .format(len(img_type)))
            
        if self._video[img_type[0]] is None:
            self._video[img_type[0]] = (
                    self.process_arbitrary(img_type[0], k[0])
                    )
            
        if self._video[img_type[1]] is None:
            self._video[img_type[1]] = (
                    self.process_arbitrary(img_type[1], k[1])
                    )
            
        self._img_type = img_type
        self.rng = [-0.01, 0.01]
        
    def change_fps(self, n):
        """
        Sums n frames into one, hence changes the frame rate of the video.
        Works only on the raw data. 
        Therefore call before calling make_... functions
        
        Parameters:
            n (int): number of integrated frames
            
        Returns:
            no return
            
        """
        out=np.ndarray(
                list(self._video['raw'].shape[0: 2])+
                [self._video['raw'].shape[2] // n - 1]
                )
        
        t_out=[]
        
        for i in range(n,self._video['raw'].shape[-1] // n * n, n):
            out[:, :, i // n - 1] = np.average(
                    self._video['raw'][:, :, i - n: i], 
                    axis = 2
                    )
            
            t_time = self.time_info[i][0]
            t_period = 0
            
            for t in self.time_info[i - n: i]:
                t_period += t[1]
                
            t_time += t_period
            t_out.append([t_time, t_period])
            
        self._video['raw'] = out
        self.time_info=t_out
        self.refresh()
                
    def image_properties(self, it = 'int', level = 20):
        f = np.fft.fft2(self._video[it][:, :, -1])
        magnitude_spectrum = 20 * np.log(np.abs(f))
        mask = np.real(magnitude_spectrum) > level     
        
        std = np.std(self.video[-20: -1, :, :])
        four_ampli = sum(magnitude_spectrum[mask])
               
        return four_ampli
        
    def fouriere(self, level = 30, show = False):
        
        img_type = []
        for it in self._img_type:
            if it == 'int' or it == 'diff':
                img_type.append(it)
       
        for it in img_type:
            print('Fourier filter {}'.format(it))
            
            for i in range(self.length):
                print('\r\t{}/ {}'.format(i + 1, self.length), end = '')   
                f = np.fft.fft2(self._video[it][:, :, i])
                magnitude_spectrum = 20 * np.log(np.abs(f))

                mask = np.real(magnitude_spectrum) > level
                f[mask] = 0
                  
                img_back = np.fft.ifft2(f)
                self._video[it][:, :, i] = np.real(img_back)
                
            print(' DONE')
        
        if show:
            fig_four, axes_four = plt.subplots()
            axes_four.imshow(
                    magnitude_spectrum, 
                    cmap = 'gray', 
                    vmin=-50, 
                    vmax=50
                    )
            
        
    def alpha_detect_nps(self, px):
        points_to_do = set()
        pixels_in_np = set()
        points_to_do.add(px)

        while len(points_to_do) != 0:
            f, y, x = points_to_do.pop()
            if (f, y, x) in pixels_in_np:
                continue
            
            found_pxs = self.mask[
                    f - 3 : f + 4, 
                    y - 1 : y + 2, 
                    x - 1 : x + 2
                    ].nonzero()

            for i in range(len(found_pxs[0])):
                points_to_do.add(
                        (
                                f+found_pxs[0][i] - 3, 
                                y+found_pxs[1][i] - 1, 
                                x+found_pxs[2][i] - 1
                                )
                        )
                
            pixels_in_np.add((f, y, x))
            
        npf = int(round(np.average([p[0] for p in pixels_in_np])))
        npy = np.average([p[1] for p in pixels_in_np])
        npx = np.average([p[2] for p in pixels_in_np])
            
        return pixels_in_np, (npf, npy, npx)
        
    def img_process_alpha(
            self, 
            threshold = 15, 
            dip = -0.003, 
            noise_level = 0.001
            ):
        
        self._img_proc_type = 'alpha'
        
        if  'diff' not in self._img_type:
            print('Processes only the differential image. '
                  'Use make_diff method first.')
            return
        
        self.threshold = threshold
        self.dip = dip
        self.noise_level = noise_level
              
        self.frame_np_ids = [[] for i in range(self.length)]
        self.candidates = set()
        
        i = 0
        time = tt.time()
        all_processes = self.shape[1]*self.shape[2]   
        skipped_corr = 0
        skipped_peak = 0
        
        for x in range(self.shape[2]):
            for y in range(self.shape[1]):
                
                if np.abs(self.video_from('diff')[:, y, x]).max() > noise_level:
                    corr_out = ahm.correlation_temporal(
                            self.video_from('diff')[:, y, x], 
                            self.k_diff, 
                            dip, 
                            threshold
                            )
                    
                    if len(corr_out['bind'][0]) != 0:
                        for f in corr_out['bind'][0]:
                            self.candidates.add((f, y, x))
                    else:
                        skipped_peak += 1
                else:
                    skipped_corr += 1
                    
                i += 1
                print('\r\t{}/ {}, remains {:.2f} s'.format(
                        i + 1, 
                        all_processes, 
                        (tt.time()-time) / i * (all_processes - i)), 
                        end = ''
                        ) 
                        
        self.px_for_image_mask = copy.deepcopy(self.candidates)
                
        print(' DONE')
        print('#PXS excluded from correlation: {} / {}, {:.1f} %'
              .format(
                      skipped_corr, 
                      all_processes, 
                      skipped_corr/all_processes * 100
                      ))
        print('#PXS excluded from peaks: {} / {}, {:.1f} %'
              .format(
                      skipped_peak, 
                      all_processes-skipped_corr, 
                      skipped_peak/(all_processes-skipped_corr) * 100
                      ))
        
        self.mask = self.process_mask()

        print('Connecting the detected pxs into patterns.', end = '')
        
        np_id = 0
        
        while len(self.candidates) != 0:
            pixels_in_np, position = self.alpha_detect_nps(
                    self.candidates.pop()
                    )
            
            nanoparticle = NanoParticle(
                    np_id, 
                    position, 
                    pixels_in_np, 
                    method = 'alpha'
                    )
            
            self.np_database.append(nanoparticle)
            
            for f in range(- self.k_diff // 2, self.k_diff // 2):
                self.frame_np_ids[position[0] + f].append(np_id)
            
            self.candidates.difference_update(pixels_in_np)
            np_id += 1

        print(' DONE')    
        
        self.show_mask = True
        self.show_pixels = True
        self.show_detected = True
        
    def beta_peaks_processing(self, px):
        positions_in_np = []
        points_excluded = set()

        f, y, x = px
        
        dist = [0] + list(self.idea3d.shape[:2])
        dist = [d//2 for d in dist]
        df, dy, dx = dist
        dys = dy
        dxs = dx

        if y < dy:
            dys = 0
        if x < dx:
            dxs = 0
        dist = (df, dys, dxs)

        neighbors_indeces = self.mask[f, y-dys:y+dy, x-dxs:x+dx].nonzero()
        neighbors = [np.array([f, y , x]) - dist + np.array(
                [0, neighbors_indeces[0][i], neighbors_indeces[1][i]]
                ) for i in range(len(neighbors_indeces[0]))]

        neighbors_values = [self.video[tuple(n)] for n in neighbors]
        central_point = neighbors[np.argmax(neighbors_values)]
        positions_in_np.append(tuple(central_point))
        for n in neighbors:
            points_excluded.add(tuple(n))

        i = 1         
        go = f + 1 < self.length
        last_point = copy.deepcopy(central_point)
        while go:
            neighbors_indeces = self.mask[
                    f + i, 
                    y - dys : y + dys, 
                    x - dx : x + dx
                    ].nonzero() 

            if len(neighbors_indeces[0]) != 0:   
                neighbors = [np.array([f + i, y , x]) - dist + np.array(
                    [0, neighbors_indeces[0][j], neighbors_indeces[1][j]]
                    ) for j in range(len(neighbors_indeces[0]))]
    
                norms = [np.linalg.norm(last_point - n) for n in neighbors]
                closest_point = neighbors[np.argmin(norms)]
                last_point = copy.deepcopy(closest_point)    
                positions_in_np.append(tuple(closest_point))
                
                for n in neighbors:
                    points_excluded.add(tuple(n))
            else:
                go = False
                
            i += 1    
            
            if f + i >= self.length:
                go = False
                           
        nanoparticle = NanoParticle(0, positions_in_np)  
                
        return nanoparticle, points_excluded
        
    def image_process_beta(self, threshold = 100):
        self.make_corr()

        self.frame_np_ids = [[] for i in range(self.length)]
        self.candidates = list()

        for f in range(self.length):
            coordinates = peak_local_max(
                    self._video['corr'][:, :, f], 
                    threshold_abs = threshold
                    )
            
            for c in coordinates:
                y, x = c
                self.candidates.append((f, y, x))
                
        self.px_for_image_mask = copy.deepcopy(self.candidates)           
        self.mask = self.process_mask()  
        
        np_id = 0
        while len(self.candidates) != 0:
            nanoparticle, points_excluded = self.beta_peaks_processing(
                    self.candidates.pop(0)
                    )
            nanoparticle.id = np_id
            
            for pe in points_excluded:
                self.mask[pe] = 0
                if pe in self.candidates:
                    self.candidates.remove(pe)
                    
            self.np_database.append(nanoparticle)
            
            for f in range(
                    nanoparticle.positions[0][0], 
                    nanoparticle.positions[-1][0]+1
                    ):
                self.frame_np_ids[f].append(np_id)
            
            np_id += 1
         
        self.show_mask = True
        self.show_pixels = True
        self.show_detected = True
        
    def gamma_peaks_processing(self, px):
        positions_in_np = []
        points_excluded = set()

        f, y, x = px
        positions_in_np.append(px)
        points_excluded.add(px)
        
        dist = [0] + list(self.idea3d.shape[:2])
        dist = [d // 2 for d in dist]
        df, dy, dx = dist
        dys = dy
        dxs = dx

        if y < dy:
            dys = y
        if x < dx:
            dxs = x
        dist = (df, dys, dxs)

        i = 1         
        go = f + 1 < self.length
        last_point = px
        
        while go:
            neighbors_indeces = self.mask[
                    f + i, 
                    y - dys : y + dy, 
                    x - dxs : x + dx
                    ].nonzero() 

            if len(neighbors_indeces[0]) != 0:   
                neighbors = [
                        np.array([f + i, y , x]) - 
                        dist + 
                        np.array([
                                0, 
                                neighbors_indeces[0][j], 
                                neighbors_indeces[1][j]
                                ]) 
                for j in range(len(neighbors_indeces[0]))
                ]
    
                norms = [np.linalg.norm(last_point - n) for n in neighbors]
                closest_point = neighbors[np.argmin(norms)]
                last_point = copy.deepcopy(closest_point)    
                positions_in_np.append(tuple(closest_point))
                points_excluded.add(tuple(closest_point))
            else:
                go = False
                
            i += 1    
            
            if f+i >= self.length:
                go = False
                
        np_masks = []        
        
        for position in positions_in_np:
            np_masks.append(self._dict_of_patterns[position])
            
        nanoparticle = NanoParticle(
                0, 
                positions_in_np, 
                masks = np_masks
                )         
        
        return nanoparticle, points_excluded
          
    def image_process_gamma(self, threshold = 100):
        self.threshold = threshold
        self.make_corr()
        
        self.frame_np_ids = [[] for i in range(self.length)]
        self.candidates = []
        self._dict_of_patterns = dict()
        
        self.mask = (self._video['corr'] > threshold)*1  
        
        #600
        minimal_area = 0
#        condition = True
        
        #650
#        minimal_area = 4
#        condition = (size[1] >= size[0])
        
        #750
#        minimal_area = 10

        number = 0
        fit_failed = 0
        omitted = 0
        
        print('Processing frames')
        for f in range(self.length):
            print('\r\t{}/ {}'.format(f + 1, self.length), end = '')
            gray = self.mask[:, :, f].astype(np.uint8)    
            th, threshed = cv2.threshold(
                    gray, 
                    100, 
                    255, 
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                    )
            
            patterns = cv2.findContours(
                    threshed, 
                    cv2.RETR_LIST, 
                    cv2.CHAIN_APPROX_NONE
                    )[-2][: -1]      
            
            for pattern in patterns:
                
                for p in pattern:
                    self.px_for_image_mask.append(
                            tuple([f] + 
                                  p[0,: : -1].tolist())
                            )
                
                if minimal_area < cv2.contourArea(pattern):
                    try:
                        ellipse = cv2.fitEllipse(pattern)
                    except:
                        fit_failed += 1
                        continue
                    
                    loc, size, angle = ellipse
                    
#                    condition = (75 < angle < 105 and size[1] > size[0])
#                    condition = (size[1] >= size[0])
                    condition = True
                    
                    if condition:
                        candidate = (
                                f, 
                                int(round(loc[1])), 
                                int(round(loc[0]))
                                )
                        self.candidates.append(candidate)
                        self._dict_of_patterns[candidate] =  pattern
                    else:
                        omitted += 1
                    number += 1
                
        self.info_add("\n--gamma--")
        self.info_add("Dots number: {}".format(number))
        self.info_add("Fit fails: {}".format(fit_failed))
#        print("Omitted: {}".format(omitted))

        self.mask = self.process_mask()  
        
        #and continues in the same way as beta
        
        np_id = 0
        while len(self.candidates) != 0:
            nanoparticle, points_excluded = self.gamma_peaks_processing(
                    self.candidates.pop(0)
                    )
            nanoparticle.id = np_id
            
            for pe in points_excluded:
                self.mask[pe] = 0
                if pe in self.candidates:
                    self.candidates.remove(pe)
                    
            self.np_database.append(nanoparticle)
            
            for f in range(
                    nanoparticle.positions[0][0], 
                    nanoparticle.positions[-1][0] + 1
                    ):
                self.frame_np_ids[f].append(np_id)
            
            np_id += 1
            
        self.gamma_time_conection()
         
        self.show_mask = True
        self.show_pixels = True
        self.show_detected = True 
        
    def gamma_time_conection(self):
        dist = [d//2 for d in self.idea3d.shape[:2]]
        last_np_id = len(self.np_database)
        
        for f in range(self.length - 2):
            ending_nps = [
                    self.np_database[np_id] 
                    for np_id in self.frame_np_ids[f] 
                    if self.np_database[np_id].last_frame == f
                    ]
            beginning_nps = [
                    self.np_database[np_id] 
                    for np_id in self.frame_np_ids[f+2] 
                    if self.np_database[np_id].first_frame == f+2]

            for enp in ending_nps:
                for bnp in beginning_nps:
                    if np.max(
                            np.abs(
                                    np.array(enp.last_position_xy()) - 
                                    np.array(bnp.first_position_xy()))
                            - np.array(dist)
                            ) < 0:
                        
                        gap_position = [tuple(
                                (
                                        np.array(enp.positions[-1]) + 
                                        np.array(bnp.positions[0])
                                        )//2
                                )]
                                
                        gap_mask = [enp.masks[-1]]
                        
                        nanoparticle = NanoParticle(
                                last_np_id, 
                                enp.positions + gap_position + bnp.positions,
                                masks = enp.masks + gap_mask + bnp.masks
                                )
                        enp.good = False
                        bnp.good - False
                        
                        for f in range(
                                nanoparticle.first_frame, 
                                nanoparticle.last_frame + 1
                                ):
                            
                            if f <= enp.last_frame:
                                try:
                                    self.frame_np_ids[f].remove(enp.id)
                                except:
                                    pass
                                
                            if f >= bnp.first_frame:
                                try:
                                    self.frame_np_ids[f].remove(bnp.id)
                                except:
                                    pass
                            self.frame_np_ids[f].append(last_np_id)
                            
                        self.np_database.append(nanoparticle)
                        last_np_id += 1

    def statistics(self):
        self.make_frame_stats()
        self.show_stats = True
        
        if len(self.np_database) != 0:
        
            self.np_count_present = [0 for i in range(self.length)]
            self.np_count_first_occurance = [0 for i in range(self.length)]
            
            for f in range(self.length):
                self.np_count_present[f] = len(self.frame_np_ids[f])
                
            already_counted = set()
            
            for f in range(self.length):
                
                for np_id in self.frame_np_ids[f]:  
                    
                    if np_id not in already_counted:
                        self.np_count_first_occurance[
                                self.np_database[np_id].first_frame
                                ] += 1
                        already_counted.add(np_id)
                                        
            self.np_count_integral = [
                    sum(self.np_count_first_occurance[: i + 1]) 
                    for i in range(len(self.np_count_first_occurance))
                    ]
            self.info_add('\n--statistics--')
            self.info_add("NP count: {}".format(self.np_count_integral[-1]))    
            
            #average binding rate
            start = 0
            end = self.length - 1
            
            for i in range(len(self.np_count_first_occurance)):
                if self.np_count_integral[i] > 10:
                    start = i
                    break
                
            for i in range(1, len(self.np_count_first_occurance)):
                if self.np_count_integral[-i] < (
                        self.np_count_integral[-1] - 
                        20
                        ):
                    end = len(self.np_count_first_occurance) - i
                    break
            
            binding_rate = (
                    self.np_count_integral[end] - 
                    self.np_count_integral[start]
                    )/(end - start) * 100
            
            self.info_add('Rate per 100 frames: {:.01f}'.format(
                    binding_rate)
            )
            
            self.info_add('Compared to reference: {:.01f} %'.format(
                    binding_rate/26*100)
            )
            
            
            np_count = self.np_count_present
    
            cut_std = self.stats_std[self.k_diff*3:]
            cut_np_count = np_count[self.k_diff*3:]  
            
            norm_std = (cut_std-min(cut_std))/max(cut_std)
            norm_np_count = (
                    np.array(cut_np_count) -
                    min(cut_np_count)
                    )/max(cut_np_count)
            
    #        difference = norm_std - norm_np_count
            self.validity = np.concatenate(
                    (np.array([0]*self.k_diff*3), 
                     np.multiply(norm_np_count, norm_std))
                    )
            
            self.valid = self.validity < 2*np.average(self.validity)
            
        else:
            self.valid = [True]*self.length
            
    def histogram(self, what = 'corr'):
        if self._video[what] is None:
            return
        
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))
        
        p0 = [1e6, 0., 50.0]

        fig, ax = plt.subplots()
        ax.set_title(self.file)
        
        n, bins, patches = ax.hist(
                np.matrix.flatten(self.video_from(what)), 
                1000, 
                color = blue
                )
        
        bin_centers = (bins[:-1] + bins[1:])/2
        
        coeff, var_matrix = curve_fit(
                gauss, 
                bin_centers, 
                n, 
                p0=p0
                )
        
        A, mu, sigma = coeff
        sigma = np.abs(sigma)
        
        span_x = ax.get_xlim()
        x = np.arange(span_x[0], span_x[1], (span_x[1] - span_x[0]) / 1e3)
        
        ax.plot(x, gauss(x, *coeff), color = red, ls = '--')
                    
        height = ax.get_ylim()[1]
        width = np.abs(span_x[1] - span_x[0]) / 1000
        
        sigma2 = mpatches.Rectangle(
                (sigma * 5, 0), 
                width, 
                height, 
                color = red
                ) 
        ax.add_patch(sigma2)

        
        sigma3 = mpatches.Rectangle(
                (sigma * 6, 0), 
                width, 
                height, 
                color = red
                )  
        ax.add_patch(sigma3)
        
        threshold = mpatches.Rectangle(
                (self.threshold, 0),
                width, 
                height, 
                color = black
                ) 
        ax.add_patch(threshold)
        
        self.info_add('\n--histogram--')
        self.info_add(
'''5 x sigma = {:.02f}
6 x sigma = {:.02f}
threshold = {}'''.format(
                5 * sigma, 
                6 * sigma,
                self.threshold
                )
                )
#    return int(round(sigma * 5))
        
    def characterize_nps(self, save = False):
        size = 25
        save_all = False
        
        for nanoparticle in self.np_database[10:15]:
            if not nanoparticle.good:
                continue
                
#            if not self.valid[nanoparticle.first_frame]:
#                continue
            all_measures = []
            all_contrast = []
            i = 0
            for np_mask in nanoparticle.masks:
                
                f = nanoparticle.first_frame + i
                y, x = nanoparticle.position_yx(f)
    
                ly = int(np.heaviside(y - size, 1)*(y - size))
                lx = int(np.heaviside(x - size, 1)*(x - size))     
                
                if y + size > self.shape[1]:
                    ry = self.shape[1]
                else:
                    ry = int(round(y + size))
                    
                if x + size > self.shape[2]:
                    rx = self.shape[2]
                else:
                    rx = int(round(x + size))
         
                raw = self.video_from('diff')[f, ly: ry, lx: rx]
            
#            npm = nanoparticle.mask_for_characterization

                
                
                img = np.zeros(raw.shape)

                
                if self._img_proc_type == 'alpha':
                    npm = np_mask
                    contour = npm - np.array([
                        [ly, lx] 
                        for i in range(npm.shape[0])
                        ])
                    print(contour)
                    img[contour[:,0], contour[:,1]] = 1
                else:
                    npm = np.squeeze(np_mask)
                    contour = npm - np.array([
                        [lx, ly] 
                        for i in range(npm.shape[0])
                        ])
                    cv2.fillPoly(img, [contour], color = 1)
                    cv2.drawContours(img, [contour], 0, 0, 1)
                mask = img == 1
          
                dy = np.max(npm[:, 1]) - np.min(npm[:, 1]) + 1
                dx = np.max(npm[:, 0]) - np.min(npm[:, 0]) + 1
                
                results = characterization.characterize(raw, mask, [dx, dy])
                all_measures.append(results)
                all_contrast.append(np.abs(results[2]))
                i += 1
                
            measures = all_measures[np.argmax(all_contrast)]

            nanoparticle.size = measures[: 2]
            nanoparticle.contrast = measures[2]
            nanoparticle.intensity = measures[4]
        
            
            if save:
                
                if not os.path.isdir(self.folder + FOLDER_EXPORTS_NP):
                    os.mkdir(self.folder + FOLDER_EXPORTS_NP)
                
                name = '{}{}/{}_info.txt'.format(
                        self.folder, 
                        FOLDER_EXPORTS_NP, 
                        self.file
                        )  
                
                if not save_all:
                    save_all = tl.before_save_file(name)
                    
                if save_all:
                    characterization.save(raw, measures, name)
                    
                else:
                    break
                         
    def exclude_nps(self, thresholds, exclude = True):
        method_lambdas = [
                lambda x: x.contrast,
                lambda x: x.size[0]
                ]
        excluded = set()
        
        for f in range(self.length):
            ids = copy.deepcopy(self.frame_np_ids[f])
            frame_excluded = set()
            
            for np_id in ids:
                
                for i in range(len(thresholds)): 
                    
                    if np_id in frame_excluded:
                        continue
                    
                    if method_lambdas[i](
                            self.np_database[np_id]
                            ) < thresholds[i]:
                        
                        if exclude:
                            self.frame_np_ids[f].remove(np_id)
                            frame_excluded.add(np_id)
                            
                            
                        else:
                            self.np_database[np_id].color = tl.hex_to_list(
                                    green
                                    )
                            frame_excluded.add(np_id)
                        self.np_database[np_id].good = False
                        excluded.add(np_id)
                
        self.info_add('\n--exclusion--')       
        self.info_add('Number of excluded nps: {}'.format(len(excluded)))
           
    def save_idea(self, name = None):
        if not os.path.isdir(self.folder + FOLDER_IDEAS):
            os.mkdir(self.folder + FOLDER_IDEAS)
            
        if name == None:
            name = self.file
            
        file_name = self.folder + FOLDER_IDEAS + '/' + name
        
        if tl.before_save_file(file_name) or name == self.file:
            
            np.save(file_name + '.npy', self.idea3d)
            np.save(file_name + '_frame' + '.npy', np.array(self._idea_frame))
            np.save(file_name + '_spanx' + '.npy', self._idea_span_x)
            np.save(file_name + '_spany' + '.npy', self._idea_span_y)
            
            print('Pattern saved')

        else:
            print('Could not save the pattern.')
        
    def load_idea(self, name = None):
        if name == None:
            name = self.file
            
        file_name = self.folder + FOLDER_IDEAS + '/' + name
        self.idea3d = np.load(file_name + '.npy')
        
        if os.path.isfile(file_name + '_frame' + '.npy'):
            self._idea_frame = np.load(file_name + '_frame' + '.npy').max()
            self._idea_span_x = np.load(file_name + '_spanx' + '.npy')
            self._idea_span_y = np.load(file_name + '_spany' + '.npy')
                
    def handle_choose_idea(self, shift, eclick, erelease):
        corner_1 = eclick.xdata, eclick.ydata
        corner_2 = erelease.xdata, erelease.ydata
        
        corner_1 = [tl.true_coordinate(b) for b in corner_1]
        corner_2 = [tl.true_coordinate(e) for e in corner_2]
        
        span_x = np.array([
                shift[1] + corner_1[0] + 1 - 2, 
                shift[1] + corner_2[0] + 2
                ])
        span_y = np.array([
                shift[0] + corner_1[1] + 1 - 2, 
                shift[0] + corner_2[1] + 2
                ])
        
        self._idea_span_x = span_x
        self._idea_span_y = span_y

        self.idea3d = self._video['diff'][
                span_y[0]: span_y[1], 
                span_x[0]: span_x[1], 
                self._idea_frame - self.k_diff: self._idea_frame + self.k_diff
                ]
        
        print('Pattern chosen')
        
    @property
    def auto_contrast(self):
        nanoparticle = None
        for np_id in self.frame_np_ids[self._idea_frame]:
            position = self.np_database[np_id].position_yx(self._idea_frame)
            
            position_x = (
                    self._idea_span_x[0] < 
                    position[1] < 
                    self._idea_span_x[1]
                    )
            
            position_y = (
                    self._idea_span_y[0] < 
                    position[0] < 
                    self._idea_span_y[1]
                    )
            
            if position_x and position_y:
                nanoparticle =  self.np_database[np_id]
                
        if nanoparticle is None:
            raise ValueError('Could not detect the initial nanoparticle')
            
        else:
            self.info_add(nanoparticle.contrast)
            return (nanoparticle.contrast - 2.5) * 0.2 + 1.6
            
    def handle_save_idea(self, event):
        self.save_idea()
        
    def explore(self, source='vid'):
                
        def frame_info(i):
            
            if len(self.np_database) !=0:
                
                if len(self.frame_np_ids[ax.index]) != 0:
                    return '{}: {}/{}  t= {} s dt= {:.2f}s\nNPs; now: {}, integral: {}'.format(
                        self._img_type[not self._toggle],
                        i,
                        ax.volume.shape[0],
                        tl.SecToMin(self.time_info[i][0]),
                        self.time_info[i][1],
                        len(self.frame_np_ids[ax.index]),
                        self.frame_np_ids[ax.index][-1],
                    )

            
            return '{}: {}/{}  t= {} s dt= {:.2f} s'.format(
                self._img_type[not self._toggle],
                i,
                ax.volume.shape[0],
                tl.SecToMin(self.time_info[i][0]),
                self.time_info[i][1]
            )

        
        def next_slice(i):
            ax.index = (ax.index + i) % ax.volume.shape[0]
            img.set_array(ax.volume[ax.index])
            
            if self.show_pixels:
                mask.set_array(volume_mask[ax.index])  
                
            if self.show_detected:                
                [p.remove() for p in reversed(ax.patches)]
#                if self._img_type[not self._toggle] == 'diff' or self._img_type[not self._toggle] == 'corr':
                for np_id in self.frame_np_ids[ax.index]:
                    p = mpatches.Circle(
                            self.np_database[np_id].position_xy(ax.index), 
                            5, 
                            color=self.np_database[np_id].color, 
                            fill = False,
                            alpha = 0.5,
                            lw = 2)
                    ax.add_patch(p)
                    self.show_detected_all = False
                    
#                elif self._img_type[not self._toggle] == 'int':
#                    for frame in self.frame_np_ids[:ax.index+1]:
#                        for np_id in frame:
#                            p = mpatches.Circle(
#                                    self.np_database[np_id].last_position_xy(), 
#                                    5, 
#                                    color=self.np_database[np_id].color, 
#                                    fill = False,
#                                    alpha = 0.5,
#                                    lw = 2)
#                            ax.add_patch(p)
                            
            if self.show_stats:
                location.xy=[ax.index, stat_plot.get_ylim()[0]]
                fig_stat.canvas.draw()
            
            if self.valid is not None:
                if not self.valid[ax.index]:
                    for s in SIDES:
                        ax.spines[s].set_color(red)
                        ax.spines[s].set_linewidth(4)
                else:
                    for s in SIDES:
                        ax.spines[s].set_color(black)
                        ax.spines[s].set_linewidth(1)   
            ax.set_title(frame_info(ax.index))
            
        def save_frame():
            """
            checks and eventually creates the folder 
            'export_image' in the folder of data
            """
            if not os.path.isdir(self.folder + FOLDER_EXPORTS):
                os.mkdir(self.folder + FOLDER_EXPORTS)

            # creates the name, appends the rigth numeb at the end

            name = '{}/{}_T{:03.0f}_dt{:03.0f}'.format(
                    self.folder+FOLDER_EXPORTS, 
                    self.file,
                    self.time_info[ax.index][0],
                    self.time_info[ax.index][1] * 100
                    )

            i = 1
            while os.path.isfile(name + '_{:02d}.png'.format(i)):
                i += 1
            name += '_{:02d}'.format(i)

            # saves the png file of the view

            fig.savefig(name + '.png', dpi=300)
            fig_stat.savefig(name + '_int.png', dpi=300)

            xlim = [int(i) for i in ax.get_xlim()]
            ylim = [int(i) for i in ax.get_ylim()]

#            saves the exact nad precise tiff file
            pilimage = Image.fromarray(img.get_array()[
                    ylim[1]: ylim[0],
                    xlim[0]: xlim[1]
                    ])
            
#            pilimage.save(name + '.tiff')
            print('File SAVED @{}'.format(name))

            img.set_array(ax.volume[ax.index])
            fig.canvas.draw_idle()
        
        def mouse_scroll(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            
            if event.button == 'down':
                next_slice(1)
                
            elif event.button == 'up':
                next_slice(-1)
                
            fig.canvas.draw()

        def mouse_click(event):
            
            if event.dblclick:
                fig = event.canvas.figure
                ax = fig.axes[0]
                x = int((event.xdata + 0.5) // 1)
                y = int((event.ydata + 0.5) // 1)

                print('------------')
                print('x = {}'.format(x))
                print('y = {}'.format(y))
                ip.correlation_temporal(
                        ax.volume[: , y, x], 
                        k_diff = self.k_diff, 
                        step = self.dip, 
                        threshold = self.threshold,  
                        show = True
                        )
                
        def button_press(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            
            if event.key == '6':
                next_slice(10)
                fig.canvas.draw()
            elif event.key == '4':
                next_slice(-10)
                fig.canvas.draw()
            elif event.key == '9':
                next_slice(100)
                fig.canvas.draw()
            elif event.key == '7':
                next_slice(-100)
                fig.canvas.draw()
            elif event.key == '3':
                next_slice(1)
                fig.canvas.draw()
            elif event.key == '1':
                next_slice(-1)
                fig.canvas.draw()
            elif event.key == 't':
                "Toggle between img types"
                ax.volume = self.video
                
                if self._img_type[not self._toggle] == 'corr':
                    img.set_clim(
                            np.min(self.video_from('corr')),
                            np.max(self.video_from('corr'))
                            )
                elif self._img_type[not self._toggle] == 'raw':
                    img.set_clim(0, 1)
                    
                else:
                    img.set_clim(self.rng[0], self.rng[1])
                    
                next_slice(0)
                fig.canvas.draw()
            
            elif event.key == 'd':
                "Define the NP pattern"
                xlim = [int(i) for i in ax.get_xlim()]
                ylim = [int(i) for i in ax.get_ylim()]
                
                self._idea_frame = ax.index

                raw_idea = img.get_array()[
                    ylim[1]: ylim[0],
                    xlim[0]: xlim[1]
                    ]
                
                def toggle_selector(event):
                    pass
            
                fig_idea, ax_idea = plt.subplots()
                img_idea = ax_idea.imshow(
                        raw_idea,
                        cmap='gray'
                        )
                
                toggle_selector.RS = RectangleSelector(
                        ax_idea, 
                        lambda eclick, erelease: self.handle_choose_idea(
                                (ylim[1], xlim[0]), 
                                eclick, 
                                erelease
                                ),
                        drawtype=  'box', useblit=True,
                        button = [1, 3],  # don't use middle button
                        minspanx = 5, 
                        minspany = 5,
                        spancoords = 'pixels',
                        interactive = True
                        )
                
                plt.connect('key_press_event', toggle_selector)
                fig_idea.canvas.mpl_connect(
                        'close_event', 
                        self.handle_save_idea
                        )
                
            elif event.key == 'm':
                "switch off/on the overlaying graphics"
                if self.show_graphic:
                    img.set_zorder(10)
                    self.show_graphic = False
                    
                else:
                    img.set_zorder(0)
                    self.show_graphic = True
                    
                fig.canvas.draw()
            elif event.key == 'b':
                
                if self.show_detected_all == False:
                    
                    for frame in self.frame_np_ids[:ax.index+1]:
                        
                        for np_id in frame:
                            p = mpatches.Circle(
                                    self.np_database[np_id].last_position_xy(), 
                                    5, 
                                    color=self.np_database[np_id].color, 
                                    fill = False,
                                    alpha = 0.5,
                                    lw = 2)
                            ax.add_patch(p)
                            
                    self.show_detected_all = True
                    
                else:
                    next_slice(0)
                    self.show_detected_all = False
                    
                    
            elif event.key == 'n':
                
                if self.show_mask:
                    mask.set_zorder(-1)
                    self.show_mask = False
                    
                else:
                    mask.set_zorder(1)
                    self.show_mask = True
                fig.canvas.draw()
                
            elif event.key == '5':
                lim = [i * 1.2 for i in img.get_clim()]
                img.set_clim(lim)
            elif event.key == '8':
                lim = [i / 1.2 for i in img.get_clim()]
                img.set_clim(lim)
            elif event.key == 'a':
                save_frame()
               
            img.set_array(ax.volume[ax.index])
            fig.canvas.draw_idle()

        fig, ax = plt.subplots()
        ax.volume = self.video
        ax.index = 0
        ax.set_title(frame_info(ax.index))

        if self._img_type[not self._toggle] == 'raw':
            img = ax.imshow(
                    ax.volume[ax.index], 
                    cmap='gray', 
                    zorder = 0
                    )
            
        elif self._img_type[not self._toggle] == 'corr':
            img = ax.imshow(
                    ax.volume[ax.index], 
                    cmap='gray', 
                    zorder = 0, 
                    vmin=np.min(self.video_from('corr')), 
                    vmax=np.max(self.video_from('corr'))
                    ) 
            
        else:
            img = ax.imshow(
                    ax.volume[ax.index], 
                    cmap='gray', 
                    zorder = 0, 
                    vmin=self.rng[0], 
                    vmax=self.rng[1]
                    )
            
        if self.show_pixels:
            volume_mask = self.process_mask_image()
            ax.volume_mask = volume_mask
            mask = ax.imshow(
                    volume_mask[ax.index]
                    )

        fig.canvas.mpl_connect('scroll_event', mouse_scroll)
        fig.canvas.mpl_connect('button_press_event', mouse_click)
        fig.canvas.mpl_connect('key_press_event', button_press)

        fontprops = fm.FontProperties(size=10)
        scalebar = AnchoredSizeBar(ax.transData,
                   34, '100 $\mu m$', 'lower right', 
                   pad=0.1,
                   color='black',
                   frameon=False,
                   size_vertical=1,
                   fontproperties=fontprops)

        ax.add_artist(scalebar)
        
        if self.show_stats:
            
            fig_stat, stat_plot = plt.subplots()
            stat_plot.grid(linestyle='--')
            stat_plot.set_title('info')
            stat_plot.set_xlabel('frame')
            stat_plot.set_ylabel('std of intensity [a. u.]')
            
            if len(self.np_database) > 0:
                np_plot = stat_plot.twinx()
                validity_plot = stat_plot.twinx()
                
                np_plot.set_ylabel('Number of NPs')
#                validity_plot.set_ylabel('Validity [Au]')
#                validity_plot.spines['right'].set_color(red)
#                validity_plot.yaxis.label.set_color(red)
#                validity_plot.tick_params(axis='y', colors=red)

            stat_plot.plot(
                    self.stats_std, 
                    linewidth=1, 
                    color=yellow, 
                    label='stdev'
                    )
            stat_plot.yaxis.label.set_color(yellow)
            stat_plot.spines['left'].set_color(yellow)
            stat_plot.tick_params(axis='y', colors=yellow)
            stat_plot.plot(
                    [np.average(self.stats_std) for i in self.stats_std], 
                    linewidth=1, 
                    color=yellow, 
                    label='average stdev', 
                    ls=':'
                    )  
            
#            stat2_plot = stat_plot.twinx()
#            stat2_plot.plot(
#                    self.stats_min, 
#                    linewidth = 1, 
#                    color = blue, 
#                    label = 'min'
#                    )
#            stat2_plot.plot(
#                    self.stats_max, 
#                    linewidth = 1, 
#                    color = blue, 
#                    label = 'max'
#                    )
#            
#            nmax = self.stats_max / np.max(self.stats_max)
#            nmin = self.stats_min / np.min(self.stats_min)
#            
#            stat2_plot.plot(
#                    nmin, 
#                    linewidth = 1, 
#                    color = blue, 
#                    label = 'min'
#                    )
#            stat2_plot.plot(
#                    nmax, 
#                    linewidth = 1, 
#                    color = blue, 
#                    label = 'max', 
#                    ls = ':'
#                    )
#            stat2_plot.plot(
#                    np.multiply(nmax, nmax), 
#                    linewidth = 1, 
#                    color = red, 
#                    label = 'max'
#                    )

            
            if len(self.np_database) > 0:              
                np_plot.yaxis.set_major_locator(MaxNLocator(integer=True))
                np_plot.plot(
                        self.np_count_present, 
                        linewidth=1, 
                        color=black, 
                        label='count in frame', 
                        ls=':'
                        )  
                np_plot.plot(
                        self.np_count_integral, 
                        linewidth=1, 
                        color=black, 
                        label='integral count'
                        )

                validity_plot.plot(
                        self.validity, 
                        color = red, 
                        label = 'validity'
                        )
                validity_plot.plot(
                        [
                                np.average(self.validity)*2 
                                for i in range(self.length)
                                ], 
                        color = red, 
                        label = 'validity, 2avg', 
                        ls=':'
                        )
                
                for tick in validity_plot.get_yticklines():
                    tick.set_visible(False)
                    
                for tick in validity_plot.get_yticklabels():
                    tick.set_visible(False)
                    
            rectangle_height = np.abs(
                    stat_plot.get_ylim()[1] - 
                    stat_plot.get_ylim()[0]
                    )
            location = mpatches.Rectangle(
                    (ax.index, stat_plot.get_ylim()[0]), 
                    1, 
                    rectangle_height, 
                    color=red
                    )                
            stat_plot.add_patch(location)
            fig_stat.legend(loc=2)
                        
                        
#        cb = fig.colorbar(img, ax=ax)
            
        plt.tight_layout()
        plt.show()

#        print('='*50)
#        print('''
#BASIC SHORTCUTS
#
#"8"/"5" increases/decreases contrast
#Mouse scrolling moves the time 
#"1" and "3" jumps 1 frames in time
#"4" and "6" jumps 10 frames in time
#"7" and "9" jumps 100 frames in time
#"f" fulscreen
#"o" zooms chosen area.
#"a" saves the image
#"s" saves the the whole figure
#"m" disables all the overlaying graphics
#"n" disables pixels recognized by correlation
#"b" shows all the detected NPs up to current frame
#"t" toggles differential and integral image, when the method "make_both" is used
#
#"Left mouse button double click" show the time/intensity point of the pixel with the correlation function.
#
#Official MATPLOTLIB shortcuts at https://matplotlib.org/users/navigation_toolbar.html
#Buttons "j"/"m" serve to increasing/decreasing contrast 
#Button "s" saves the current image as tiff file
#Mouse scrolling moves to neighboring frames
#Official shortcuts here https://matplotlib.org/users/navigation_toolbar.html
#Right mouse button click selects and switches to analysis of chosen NP image
#Double click plots the intensity course of the pixel and decides if it includes NP
#              ''')
