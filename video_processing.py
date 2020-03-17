import numpy as np
import math as m
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from PIL import Image
import cv2
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import copy
import time as tt
from skimage.feature import peak_local_max

from np_analysis import np_analysis, is_np, measure_new, visualize_and_save
import image_processing as ip
import tools as tl
from nanoparticle import NanoParticle

FOLDER_NAME = '/exports'
yellow='#ffb200'
red='#DD5544'
blue='#0284C0'
black='#000000'    
SIDES = ['left', 'right', 'bottom', 'top']
            
class Video(object):

    def __init__(self, folder, file):
        self.folder = folder
        self.file = file
        self.file_name = folder + file
        self.video_stats = None
        self.length = None
        
        self._video = {
                'raw': None,
                'diff': None,
                'int': None
                }
        self._toggle = True
        self._img_type = 'raw'
        
        self.view = None
        self.rng = [-1, 1]
        self.time_info = None

        self.np_number=0
        self.ref_frame=0
        self.k_diff = None
        self.k_int = None
        
        self.mask = None        
        self.candidates = None
        self.px_for_image_mask = None

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
        self.idea3d = None
        
        
        #show
        self.show_graphic = True
        self.show_pixels = False
        self.show_detected = False
        self.show_detected_all = False
        self.show_stats = False
        self.show_mask = True
        
    def __iter__(self):
        self.n = -1
        self.MAX = self.video.shape[2] - 1
        return self

    def __next__(self):
        if self.n < self.MAX:
            self.n += 1
            return self.video[:, :, self.n]
        else:
            raise StopIteration

    @property
    def video(self):
        if self._img_type == True:
            self._img_type = False
            return np.swapaxes(np.swapaxes(self._video['int'], 0, 2), 1, 2)
        elif self._img_type == False:
            self._img_type = True
            return np.swapaxes(np.swapaxes(self._video['diff'], 0, 2), 1, 2)
        else:
            return np.swapaxes(np.swapaxes(self._video[self._img_type], 0, 2), 1, 2)

    def loadData(self):
        self.video_stats = self.loadBinVideoStats()
        self.length = self.video_stats[1][2]
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
        self.view = [0, 0, video_width, video_hight]
        return [video_fps, [video_width, video_hight, video_length]]

    def loadBinVideo(self):
        code_format = np.float64  # type double
        suffix = '.bin'
        with open(self.file_name + suffix, mode='rb') as fid:
            video = np.fromfile(fid, dtype=code_format)
            fid.close()
            
        video = np.reshape(video, (self.video_stats[1][0],
                                   self.video_stats[1][1],
                                   self.video_stats[1][2]), order='F')

        return np.swapaxes(video, 0, 1)

    def process_diff(self, k = 1):
        sh = self._video['raw'].shape
        out = np.zeros(sh)
        
#        print((sh[0], sh[1], k))
        out[:, :, :2*k] = np.zeros((sh[0], sh[1], 2*k))
        print('Computing the differential image')
        
        for i in range(2*k, sh[-1]):
            
            
            print('\r\t{}/ {}'.format(i+1, self.video_stats[1][2]), end="")
            current = np.sum(self._video['raw'][:,:,i - k+1: i+1], axis=2)/k
            previous = np.sum(self._video['raw'][:,:,i - 2*k+1: i - k+1], axis=2)/k
#            difference = current - previous
#            average = np.average(difference)
#            print(average)
#            out[:, :, i] = difference - np.full(difference.shape, average)
            out[:, :, i] = current - previous
        self.k_diff = k
        
        print(' DONE')
        return out
    
    def process_int(self, k = 1):
        sh = self._video['raw'].shape
        out = np.zeros(sh)
        out[:, :, 0] = np.zeros(sh[0:2])
        reference = np.sum(self._video['raw'][:,:,self.ref_frame: self.ref_frame + k], axis=2)/k
        
        print('Computing the integral img')
        
        for i in range(1, sh[-1]):
            print('\r\t{}/ {}'.format(i+1, self.video_stats[1][2]), end="")
            out[:, :, i] = self._video['raw'][:, :, i] - reference
        self.k_int = k
            
        print(' DONE')
        return out
    
    def process_mask_image(self):
        volume_mask = np.zeros(list(self.video.shape) + [4])
        k_diff = self.k_diff
        k_diff = 1
        tri = [ip.func_tri(i, k_diff, 0.5, k_diff) for i in range(int(k_diff*2))]
        for pm in self.px_for_image_mask:
            f, y, x = pm
            volume_mask[f, y, x, 1] = 1
            volume_mask[f, y, x, 3] = 1
        return volume_mask
        
        
        for pm in self.px_for_image_mask:
            f, y, x = pm
            if f+k_diff > self.video.shape[0]:
                end = self.video.shape[0]
            else:
                end = f+k_diff
            volume_mask[f-k_diff:end, y, x, 1] = [1]*(end-f+k_diff)
            volume_mask[f-k_diff:end, y, x, 3] = tri[:end-f+k_diff]
            if f+k_diff > self.video.shape[0]:
                end = self.video.shape[0]
            else:
                end = f+k_diff
            volume_mask[f-k_diff:end, y, x, 0] = [1]*(end-f+k_diff)
            volume_mask[f-k_diff:end, y, x, 3] = tri[:end-f+k_diff]
            
        return volume_mask
    
    def process_mask(self):
        volume_mask = np.zeros(self.video.shape)

        for c in self.candidates:
            volume_mask[c] = 1

        return volume_mask
    
    def process_frame_stat(self):      
        i = 0
        out = []
        print('Computing the statistics')
        for v in self.video:
            out.append(np.std(v))
            i += 1
            print('\r\t{}/ {}'.format(i+1, len(self.video[:, 1, 1])), end="")
        print(' DONE')
        return np.array(out)
    
    def make_frame_stats(self):
        self.stats_std = self.process_frame_stat()
        self.show_stats = True
        
    def make_diff(self, k = 1):
        self._video['diff'] = self.process_diff(k)
        self._img_type = 'diff'
        self.rng = [-0.01, 0.01]

    def make_int(self, k = 1):
        self._video['int']= self.process_int(k)
        self._img_type = 'int'
        self.rng = [-0.01, 0.01]
        
    def make_toggle(self, kd=1, ki=1):
        if self._video['diff'] is None and self.k_diff==kd:
            self._video['diff'] = self.process_diff(kd)
            
        if self._video['int'] is None and (self.k_int ==ki or self.k_int is None):    
            self._video['int'] = self.process_int(ki)
            
        self._img_type = False
        self.rng = [-0.01, 0.01]
        
    def change_fps(self, n):
        """
        Sums n frames into one, hence changes the frame rate of the video.
        Works only on the raw data. Therefore call before calling make_... functions
        
        Parameters:
            n (int): number of integrated frames
            
        Returns:
            no return
            
        """

        out=np.ndarray(list(self._video['raw'].shape[0:2])+[self._video['raw'].shape[2]//n-1])
        t_out=[]
#        self.make_diff()
        for i in range(n,self._video['raw'].shape[-1]//n*n,n):
#            out[:,:,i//n-1]=np.sum(self._video['raw'][:,:,i-n: i], axis=2)/n
            
#            weights_std = [np.std(self._video['diff'][:,:,i - n + j]) for j in range(n)]
#            weights_std = [w/sum(weights_std) for w in weights_std]
#            print(weights_std)
            
#            out[:,:,i//n-1]=np.average(self._video['raw'][:,:,i-n: i], axis = 2, weights = weights_std)
            out[:,:,i//n-1]=np.average(self._video['raw'][:,:,i-n: i], axis = 2)
#            out[:,:,i//n-1]=np.median(self._video['raw'][:,:,i-n: i], axis=2)
            t_time=self.time_info[i][0]
            t_period=0
            for t in self.time_info[i-n: i]:
                t_period+=t[1]
            t_time+=t_period
            t_out.append([t_time, t_period])
        self._video['raw'] = out
        self.time_info=t_out
        self.refresh()
        
    def refresh(self):
        self.video_stats[1] = [self._video['raw'].shape[1], self._video['raw'].shape[0], self._video['raw'].shape[2]]
        self.length = self.video_stats[1][2]

    def time_fouriere(self):
        middle = int(self._video.shape[2] / 2)
        out = np.zeros(self._video.shape)
        for i in range(self._video.shape[0]):
            print('done {}/{}'.format(i, self._video.shape[0]))
            for j in range(self._video.shape[1]):
                signal = self._video[i, j, :]
                fspec = np.fft.fft(signal)
                fspec[middle - 5:middle + 5] = 0

                out[i, j, :] = np.fft.ifft(fspec)
            if not self.show_original:
                self._video_new = out
            else:
                self._video = out
                
    def image_properties(self, it = 'int', level = 20):
        f = np.fft.fft2(self._video[it][:, :, -1])
        magnitude_spectrum = 20 * np.log(np.abs(f))
        mask = np.real(magnitude_spectrum) > level     
        
        std = np.std(self.video[-20:-1,:,:])
        four_ampli = sum(magnitude_spectrum[mask])
               
        return four_ampli
        
        
    def fouriere(self, level = 30, show = False):
        print('Filtering fouriere frequencies')
        if type(self._img_type) == bool:
            img_type = ['int']
        else:
            img_type = [self._img_type]
        for it in img_type:
            for i in range(self._video[it].shape[2]):

                print('\r\t{}/ {}'.format(i+1, self.video_stats[1][2]), end="")    
                f = np.fft.fft2(self._video[it][:, :, i])

                magnitude_spectrum = 20 * np.log(np.abs(f))
 
                mask = np.real(magnitude_spectrum) > level
                f[mask] = 0
                  
                img_back = np.fft.ifft2(f)
                self._video[it][:, :, i] = np.real(img_back)
        print(' DONE')
        
        if show:
            fig_four, axes_four = plt.subplots()
            axes_four.imshow(magnitude_spectrum, cmap = 'gray', vmin=-50, vmax=50)
            
        
    def alpha_detect_nps(self, px):
        points_to_do = set()
        pixels_in_np = set()
        points_to_do.add(px)

        while len(points_to_do) != 0:
            f, y, x = points_to_do.pop()
            if (f, y, x) in pixels_in_np:
                continue
            
            found_pxs = self.mask[f-3:f+4, y-1:y+2, x-1:x+2].nonzero()

            for i in range(len(found_pxs[0])):
                points_to_do.add((f+found_pxs[0][i]-3, y+found_pxs[1][i]-1, x+found_pxs[2][i]-1))
                
            pixels_in_np.add((f, y, x))
            
        npf=int(round(np.average([p[0] for p in pixels_in_np])))
        npy=np.average([p[1] for p in pixels_in_np])
        npx=np.average([p[2] for p in pixels_in_np])
            
        return pixels_in_np, (npf, npy, npx)
        
    def img_process_alpha(self, threshold = 15, dip = -0.003, noise_level = 0.001):
        if self._img_type != 'diff':
            print('Processes only the differential image. Use make_diff method first.')
            return
        
        self.threshold = threshold
        self.dip = dip
        self.noise_level = noise_level
              
        self.frame_np_ids = [[] for i in range(self.length)]
        self.candidates = set()
        
        i = 0
        time = tt.time()
        all_processes = self.video.shape[1]*self.video.shape[2]   
        skipped_corr = 0
        skipped_peak = 0
        
        for x in range(self.video.shape[2]):
            for y in range(self.video.shape[1]):
                
                if np.abs(self.video[:, y, x]).max() > noise_level:
                    corr_out = ip.correlation_temporal(self.video[:, y, x], self.k_diff, dip, threshold)
                    
                    if len(corr_out['bind'][0]) != 0:
                        for f in corr_out['bind'][0]:
                            self.candidates.add((f, y, x))
                    else:
                        skipped_peak+=1
                else:
                    skipped_corr+=1
                    
                i+=1
                print('\r\t{}/ {}, remains {:.2f} s'.format(i+1, all_processes, (tt.time()-time)/i*(all_processes-i)), end="") 
                
        print(' DONE')
        print('#PXS excluded from correlation: {} / {}, {:.1f} %'.format(skipped_corr, all_processes, skipped_corr/all_processes*100))
        print('#PXS excluded from peaks: {} / {}, {:.1f} %'.format(skipped_peak, all_processes-skipped_corr, skipped_peak/(all_processes-skipped_corr)*100))
        
        self.mask = self.process_mask(self.candidates)

        print('Connecting the detected pxs into patterns.', end="")
        np_id = 0
        while len(self.candidates) != 0:
            pixels_in_np, position = self.alpha_detect_nps(self.candidates.pop())
            
            nanoparticle = NanoParticle(np_id, position, pixels_in_np, method = 'alpha')
            self.np_database.append(nanoparticle)
            
            for f in range(-self.k_diff//2, self.k_diff//2):
                self.frame_np_ids[position[0]+f].append(np_id)
            
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
            neighbors_indeces = self.mask[f + i, y-dys:y+dys, x-dx:x+dx].nonzero() 

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
            if f+i >= self.length:
                go = False
                           
        nanoparticle = NanoParticle(0, positions_in_np)  
                
        return nanoparticle, points_excluded
        
    def image_process_beta(self, threshold = 100):
        self.idea3d = self._video['diff'][125: 131, 100: 110, 103: 143] #750    proc pres 20 framu???
#        self.idea3d = self._video['diff'][70: 73, 169:187] #750    proc pres 20 framu???
#        self.idea3d = self._video['diff'][100:104, 55:69, 59: 79] #750  raw_07
#        self.idea3d = self._video['diff'][49:53, 180:194, 41:61] #750  raw_27
#        self.idea3d = self._video['diff'][19: 23, 10: 25, 101: 121] #750  raw_27
#        self.idea3d = self._video['diff'][ 96: 100, 138: 142, 81: 111] #600

        self._video['corr'] = sc.signal.correlate(self._video['diff'], self.idea3d)*1e5
        self._img_type = 'corr'
        self._video['corr'] = self._video['corr'][:, :, :-self.idea3d.shape[2]+1]

        self.frame_np_ids = [[] for i in range(self.length)]
        self.candidates = list()

        for f in range(self.length):
            coordinates = peak_local_max(self._video['corr'][:, :, f], threshold_abs = threshold)
            for c in coordinates:
                y, x = c
                self.candidates.append((f, y, x))
                
        self.px_for_image_mask = copy.deepcopy(self.candidates)           
        self.mask = self.process_mask()  
        
        np_id = 0
        while len(self.candidates) != 0:
            nanoparticle, points_excluded = self.beta_peaks_processing(self.candidates.pop(0))
            nanoparticle.id = np_id
            
            for pe in points_excluded:
                self.mask[pe] = 0
                if pe in self.candidates:
                    self.candidates.remove(pe)
                    
            self.np_database.append(nanoparticle)
            
            for f in range(nanoparticle.positions[0][0], nanoparticle.positions[-1][0]+1):
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
        dist = [d//2 for d in dist]
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
            neighbors_indeces = self.mask[f + i, y-dys:y+dy, x-dxs:x+dx].nonzero() 

            if len(neighbors_indeces[0]) != 0:   
                neighbors = [np.array([f + i, y , x]) - dist + np.array(
                    [0, neighbors_indeces[0][j], neighbors_indeces[1][j]]
                    ) for j in range(len(neighbors_indeces[0]))]
    
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
        peak = np.argmax([self.video[position] for position in positions_in_np])
        nanoparticle = NanoParticle(0, positions_in_np, peak = positions_in_np[0][0] + peak, masks = np_masks)  
                
        return nanoparticle, points_excluded
      
    def image_process_gamma(self, threshold = 100):
        self.idea3d = self._video['diff'][125: 131, 100: 110, 103: 143] #750    proc pres 20 framu???
#        self.idea3d = self._video['diff'][70: 73, 169:187] #750    proc pres 20 framu???
#        self.idea3d = self._video['diff'][100:104, 55:69, 59: 79] #750  raw_07
#        self.idea3d = self._video['diff'][49:53, 180:194, 41:61] #750  raw_27
#        self.idea3d = self._video['diff'][19: 23, 10: 25, 101: 121] #750  raw_27
#        self.idea3d = self._video['diff'][ 96: 100, 138: 142, 81: 111] #600

        self._video['corr'] = sc.signal.correlate(self._video['diff'], self.idea3d)*1e5
        self._img_type = 'corr'
        self._video['corr'] = self._video['corr'][:, :, :-self.idea3d.shape[2]+1]

        self.frame_np_ids = [[] for i in range(self.length)]
        self.candidates = []
        self._dict_of_patterns = dict()
        
        self.mask = (np.abs(self._video['corr']) > threshold)*1  
        minimal_area = 10
        
        number = 0
        fit_failed = 0
        omitted = 0
        
        for f in range(self.length):
            gray = self.mask[:, :, f].astype(np.uint8)    
            th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            patterns = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2][:-1]      
        
            for pattern in patterns:
                if minimal_area < cv2.contourArea(pattern):
                    try:
                        ellipse = cv2.fitEllipse(pattern)
                    except:
                        fit_failed += 1
                        continue
                    
                    loc, size, angle = ellipse
                    if 80 < angle < 100 and size[1] > size[0]:
                        candidate = (f, int(round(loc[1])), int(round(loc[0])))
                        self.candidates.append(candidate)
                        self._dict_of_patterns[candidate] = pattern
                    else:
                        omitted += 1
                    number += 1
                    
        self.px_for_image_mask = copy.deepcopy(self.candidates)           

        print("Dots number: {}".format(number))
        print("Fit fails: {}".format(fit_failed))
        print("Omitted: {}".format(omitted))

        self.mask = self.process_mask()  
        
        #and continues in the same way as beta
        
        np_id = 0
        while len(self.candidates) != 0:
            nanoparticle, points_excluded = self.gamma_peaks_processing(self.candidates.pop(0))
            nanoparticle.id = np_id
            
            for pe in points_excluded:
                self.mask[pe] = 0
                if pe in self.candidates:
                    self.candidates.remove(pe)
                    
            self.np_database.append(nanoparticle)
            
            for f in range(nanoparticle.positions[0][0], nanoparticle.positions[-1][0]+1):
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
                                    np.array(enp.last_position_yx()) - 
                                    np.array(bnp.first_position_yx()))
                            - np.array(dist)
                            ) < 0:
                        
                        gap_position = [tuple(
                                (np.array(enp.positions[-1]) + np.array(bnp.positions[0]))//2
                                )]
                        gap_mask = [enp.masks[-1]]
                        
                        nanoparticle = NanoParticle(
                                last_np_id, 
                                enp.positions + gap_position + bnp.positions, 
                                masks = enp.masks + gap_mask + bnp.masks
                                )
                        
                        for f in range(nanoparticle.first_frame, nanoparticle.last_frame+1):
                            if f <= enp.last_frame:
                                self.frame_np_ids[f].remove(enp.id)
                            if f >= bnp.first_frame:
                                self.frame_np_ids[f].remove(bnp.id)
                            self.frame_np_ids[f].append(last_np_id)
                            
                        self.np_database.append(nanoparticle)
                        last_np_id += 1

    def recognition_statistics(self):
        self.make_frame_stats()
        self.show_stats = True
        
        if len(self.np_database) != 0:
        
            self.np_count_present = [0 for i in range(self.length)]
            self.np_count_first_occurance = [0 for i in range(self.length)]
            
            for f in range(self.length):
                self.np_count_present[f] = len(self.frame_np_ids[f])
                
            for nanoparticle in self.np_database:
                self.np_count_first_occurance[nanoparticle.first_frame]+=1
                
            self.np_count_integral = [sum(self.np_count_first_occurance[:i+1]) for i in range(len(self.np_count_first_occurance))]
                
            np_count = self.np_count_present
    
            cut_std = self.stats_std[self.k_diff*3:]
            cut_np_count = np_count[self.k_diff*3:]  
            
            norm_std = (cut_std-min(cut_std))/max(cut_std)
            norm_np_count = (np.array(cut_np_count)-min(cut_np_count))/max(cut_np_count)
            
    #        difference = norm_std - norm_np_count
            self.validity = np.concatenate((np.array([0]*self.k_diff*3), np.multiply(norm_np_count, norm_std)))
            
            self.valid = self.validity < 2*np.average(self.validity)
            
        else:
            self.valid = [True]*self.length
        
    def characterize_nps(self):
        
        for nanoparticle in self.np_database:
            f = nanoparticle.peak
            y, x = nanoparticle.position_yx(f)
#        for npl in self.np_detected_info:
#            f, y, x = npl[1]   
            
            ry = int(np.heaviside(y - 25, 1)*(y - 25))
            rx = int(np.heaviside(x - 25, 1)*(x - 25))
            raw = self.video[f, ry:y + 25, rx: x + 25]
            mask = np.full((raw.shape), False, dtype=bool)
            
            px_y = [y]*2
            px_x = [x]*2
            extreme_pxs = [px_y, px_x]
            
            for px in nanoparticle.mask_for_characterization:
                i = 1
                for epx in extreme_pxs:
                    if epx[0] <= px[i] <= epx[1]:
                        pass
                    else:
                        if  px[i] < epx[0]:
                            epx[0] = px[i]
                        elif  px[i] > epx[1]:
                            epx[1] = px[i]
                    i += 1
    
                my = px[1]  - ry
                mx = px[2]  - rx
#                my = px[1] - y + ry
#                mx = px[2] - x + rx
                mask[my, mx] = True
                
            dy = extreme_pxs[0][1]-extreme_pxs[0][0]+1           
            dx = extreme_pxs[1][1]-extreme_pxs[1][0]+1
                        
            measures = measure_new(raw, mask, [dx, dy])
            visualize_and_save(raw, measures, self.folder, self.file)
        
    def explore(self, source='vid'):
        

        def frame_info(i):
            if len(self.np_database) !=0:
                if len(self.frame_np_ids[ax.index]) != 0:
                    return '{}/{}  t= {} s dt= {:.2f} s\nNPs; now: {}, integral: {}'.format(
                        i,
                        ax.volume.shape[0],
                        tl.SecToMin(self.time_info[i][0]),
                        self.time_info[i][1],
                        len(self.frame_np_ids[ax.index]),
                        self.frame_np_ids[ax.index][-1],
                    )

            
            return '{}/{}  t= {} s dt= {:.2f} s'.format(
                i,
                ax.volume.shape[0],
                tl.SecToMin(self.time_info[i][0]),
                self.time_info[i][1]
            )

        def mouse_scroll(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.button == 'down':
                next_slice(1)
            elif event.button == 'up':
                next_slice(-1)
            fig.canvas.draw()

        def mouse_click(event):
            if event.button == 3:
                self.np_number+=1
                print(self.np_number)
                fig = event.canvas.figure
                ax = fig.axes[0]
                x = int(event.xdata)
                y = int(event.ydata)
                raw = ax.volume[ax.index]
                np_analysis(raw[y - 25: y + 25, x - 25:x + 25], self.folder, self.file)

                p = mpatches.Rectangle((x - 0.5, y - 0.5), 5, 5, color='#FF0000', alpha=0.5)
                ax.add_patch(p)
                print('you pressed', event.button, event.xdata, event.ydata)
                fig.canvas.draw()
                
            elif event.dblclick:
                fig = event.canvas.figure
                ax = fig.axes[0]
                x = int((event.xdata + 0.5) // 1)
                y = int((event.ydata + 0.5) // 1)
                #                file = open('data.txt', 'a')
                #                file.write('['+', '.join([str(i) for i in self._video[y, x,:]])+'],\n')
                #                file.close()
                print('------------')
                print('x = {}'.format(x))
                print('y = {}'.format(y))
                ip.correlation_temporal(ax.volume[:, y, x], k_diff=self.k_diff, step=self.dip, threshold=self.threshold,  show=True)
                
        def next_slice(i):
            ax.index = (ax.index + i) % ax.volume.shape[0]
            img.set_array(ax.volume[ax.index])
            
            if self.show_pixels:
                mask.set_array(volume_mask[ax.index])  
                
            if self.show_detected:
                [p.remove() for p in reversed(ax.patches)]
                if self._img_type == 'diff' or self._img_type or self._img_type == 'corr':
                    for np_id in self.frame_np_ids[ax.index]:
                        p = mpatches.Circle(
                                self.np_database[np_id].position_yx(ax.index), 
                                5, 
                                color=self.np_database[np_id].color, 
                                fill = False,
                                alpha = 0.5,
                                lw = 2)
                        ax.add_patch(p)
                    self.show_detected_all = False
                    
                elif self._img_type == 'int' or not self._img_type:
                    for frame in self.frame_np_ids[:ax.index+1]:
                        for np_id in frame:
                            p = mpatches.Circle(
                                    self.np_database[np_id].last_position_yx(), 
                                    5, 
                                    color=self.np_database[np_id].color, 
                                    fill = False,
                                    alpha = 0.5,
                                    lw = 2)
                            ax.add_patch(p)
                            
            if self.show_stats:
                location.xy=[ax.index, -1]
                fig_stat.canvas.draw()
            
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
             # checks and eventually creates the folder 'export_image' in the folder of data
            if not os.path.isdir(self.folder + FOLDER_NAME):
                os.mkdir(self.folder + FOLDER_NAME)

            # creates the name, appends the rigth numeb at the end

            name = '{}/{}_T{:03.0f}_dt{:03.0f}'.format(self.folder+FOLDER_NAME, self.file,
                                                                  self.time_info[ax.index][0],
                                                                  self.time_info[ax.index][1] * 100)

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
            pilimage = Image.fromarray(img.get_array()[ylim[1]:ylim[0], xlim[0]:xlim[1]])
            pilimage.save(name + '.tiff')
            print('File SAVED @{}'.format(name))

            img.set_array(ax.volume[ax.index])
            fig.canvas.draw_idle()
            
        def button_press(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.key == '6':
                fig = event.canvas.figure
                next_slice(10)
                fig.canvas.draw()
            elif event.key == '4':
                fig = event.canvas.figure
                next_slice(-10)
                fig.canvas.draw()
            elif event.key == '9':
                fig = event.canvas.figure
                next_slice(100)
                fig.canvas.draw()
            elif event.key == '7':
                fig = event.canvas.figure
                next_slice(-100)
                fig.canvas.draw()
            elif event.key == '3':
                fig = event.canvas.figure
                next_slice(1)
                fig.canvas.draw()
            elif event.key == '1':
                fig = event.canvas.figure
                next_slice(-1)
                fig.canvas.draw()
            elif event.key == 'x':
                [p.remove() for p in reversed(ax.patches)]
            elif event.key == 't':
                ax.volume = self.video
                next_slice(0)
                fig.canvas.draw()
            elif event.key == 'm':
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
                                    self.np_database[np_id].last_position_yx(), 
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
            elif event.key == 'p':
                self.np_number=0
            elif event.key == 'a':
                # checks and eventually creates the folder 'export_image' in the folder of data
                if not os.path.isdir(self.folder + FOLDER_NAME):
                    os.mkdir(self.folder + FOLDER_NAME)

                # creates the name, appends the rigth numeb at the end

                name = '{}/{}_T{:03.0f}_dt{:03.0f}'.format(self.folder+FOLDER_NAME, self.file,
                                                                      self.time_info[ax.index][0],
                                                                      self.time_info[ax.index][1] * 100)

                i = 1
                while os.path.isfile(name + '_{:02d}.png'.format(i)):
                    i += 1
                name += '_{:02d}'.format(i)

                # saves the png file of the view

                fig.savefig(name + '.png', dpi=300)
                fig_stat.savefig(name + '_int.png', dpi=300)

                xlim = [int(i) for i in ax.get_xlim()]
                ylim = [int(i) for i in ax.get_ylim()]

                # saves the exact nad precise tiff file
#                pilimage = Image.fromarray(img.get_array()[ylim[1]:ylim[0], xlim[0]:xlim[1]])
#                pilimage.save(name + '.tiff')
                print('File SAVED @{}'.format(name))

            img.set_array(ax.volume[ax.index])
            fig.canvas.draw_idle()


        fig, ax = plt.subplots()
        ax.volume = self.video
        ax.index = 0
        ax.set_title('{}/{}  t= {:.2f} s dt= {:.2f} s'.format(ax.index, ax.volume.shape[0], self.time_info[ax.index][0],
                                                              self.time_info[ax.index][1]))

        if self._img_type == 'raw':
            img = ax.imshow(ax.volume[ax.index], cmap='gray', zorder = 0)
        elif self._img_type == 'corr':
            img = ax.imshow(ax.volume[ax.index], cmap='gray', zorder = 0, vmin=np.min(self.video), vmax=np.max(self.video)) 
        else:
            img = ax.imshow(ax.volume[ax.index], cmap='gray', zorder = 0, vmin=self.rng[0], vmax=self.rng[1])
            
        if self.show_pixels:
            volume_mask = self.process_mask_image()
            ax.volume_mask = volume_mask
            mask = ax.imshow(volume_mask[ax.index])
            
                       
            
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
            stat_plot.set_xlabel('time [min]')
            stat_plot.set_ylabel('std of intensity [a. u.]')
            
            if len(self.np_database) > 0:
                np_plot = stat_plot.twinx()
                validity_plot = stat_plot.twinx()
                
                np_plot.set_ylabel('Number of NPs')
                validity_plot.set_ylabel('Validity [Au]')
                validity_plot.spines['right'].set_color(red)
                validity_plot.yaxis.label.set_color(red)
                validity_plot.tick_params(axis='y', colors=red)

            stat_plot.plot(self.stats_std, linewidth=1, color=yellow, label='stdev')
            stat_plot.plot([np.average(self.stats_std) for i in self.stats_std], linewidth=1, color=yellow, label='average stdev', ls=':')  
#            stat_plot.set_ylim((0, 0.003))
            
            if len(self.np_database) > 0:              
                np_plot.yaxis.set_major_locator(MaxNLocator(integer=True))
                np_plot.plot(self.np_count_present, linewidth=1, color=black, label='integral count', ls=':')  
                np_plot.plot(self.np_count_integral, linewidth=1, color=black, label='count in frame')

                validity_plot.plot(self.validity, color = red, label = 'validity')
                validity_plot.plot([np.average(self.validity)*2 for i in range(self.length)], color = red, label = 'validity, 2avg', ls=':')
                
            rectangle_height = np.abs(stat_plot.get_ylim()[1] - stat_plot.get_ylim()[0])
            location = mpatches.Rectangle((ax.index, -1), 1/60, rectangle_height, color=red)                
            stat_plot.add_patch(location)
            fig_stat.legend(loc=2)
                        
                        
#        cb = fig.colorbar(img, ax=ax)
            
#        info = 'ej\nneki tekst\n i još nešta'
#        ax.text(0, self.video.size+20, info, fontsize=10, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
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
