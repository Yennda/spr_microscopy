import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from PIL import Image
import cv2
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import image_processing as ip
from multiprocessing import Pool
import time as tt
from skimage.feature import peak_local_max

from np_analysis import np_analysis, is_np, measure_new, visualize_and_save
import tools as tl

FOLDER_NAME = '/exports'
yellow='#ffb200'
red='#DD5544'
blue='#0284C0'
black='#000000'    
            
class Video(object):

    def __init__(self, folder, file):
        self.folder = folder
        self.file = file
        self.file_name = folder + file
        self.video_stats = None
        
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
        
        self.frames_binding = None
        self.frames_unbinding = None
        self.intensity_binding = None
        self.intensity_unbinding = None
        self.mask = None
        self.candidate = set()
        self.np_amount = 0
        self.np_marks_positions = None
        self.np_positions = []
        #[set of pxs, (npf, npy, npx), peak value, correlation]
        self.np_detected_info = []
        self.stats_std = []
        
        self.threshold = 4
        self.dip = -0.003
        self.noise_level = 0.001
        
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
        tri = [ip.func_tri(i, k_diff, 0.5, k_diff) for i in range(int(k_diff*2))]
        
        i = 1
        imax = self.video.shape[2]*self.video.shape[1]
        print('Computing the px visualisation')
        for x in range(self.video.shape[2]):
            for y in range(self.video.shape[1]):
                for f in self.frames_binding[x][y]:
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 1] = [1]*2*k_diff
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 3] = tri
                    
                for f in self.frames_unbinding[x][y]:
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 0] = [1]*2*k_diff
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 3] = tri
                i += 1
                print('\r\t{}/ {}'.format(i+1, imax), end="")
        print(' DONE')
        return volume_mask
    
    def process_mask(self):
        volume_mask = np.zeros(self.video.shape)
        
        i = 0
        imax = self.video.shape[2]*self.video.shape[1]
        
        print('Computing the y/n mask')
        for x in range(self.video.shape[2]):
            for y in range(self.video.shape[1]):
                for f in self.frames_binding[x][y]:
                    volume_mask[f, y, x] = 1
                i += 1
                print('\r\t{}/ {}'.format(i+1, imax), end="")
        print(' DONE')
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
        return out
    
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
    def characterize(self, it = 'int', level = 20):
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
            
        
    def detect_nps(self, px):
        points_to_do = set()
        points_done = set()
        points_to_do.add(px)

        while len(points_to_do) != 0:
            f, y, x = points_to_do.pop()
            if (f, y, x) in points_done:
                continue
            
            found_pxs = self.mask[f-3:f+4, y-1:y+2, x-1:x+2].nonzero()

            for i in range(len(found_pxs[0])):
                points_to_do.add((f+found_pxs[0][i]-3, y+found_pxs[1][i]-1, x+found_pxs[2][i]-1))
                
            points_done.add((f, y, x))
            
        npf=int(round(np.average([p[0] for p in points_done])))
        npy=np.average([p[1] for p in points_done])
        npx=np.average([p[2] for p in points_done])
        
        self.np_positions[npf][0].append(npx)    
        self.np_positions[npf][1].append(npy) 
        
#        for i in range(-self.k_diff//2, self.k_diff//2):
        for i in range(1):
            self.np_marks_positions[npf+i][0].append(npx)    
            self.np_marks_positions[npf+i][1].append(npy)
            
        self.np_detected_info.append([points_done, (int(round(npf)), int(round(npy)), int(round(npx)))])
        return points_done
        
    def img_process_alpha(self, threshold = 15, dip = -0.003, noise_level = 0.001):
        if self._img_type != 'diff':
            print('Processes only differential image. Use make_diff method first.')
            return
        
        print('Correlation')
        self.threshold = threshold
        self.dip = dip
        self.noise_level = noise_level
        
        self.frames_binding = [[[] for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]
        self.frames_unbinding = [[[] for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]
#        self.intensity_binding = [[0 for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]
#        self.intensity_unbinding = [[0 for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]
        self.intensity_binding = []
        self.intensity_unbinding = []
        
        i = 0
        time = tt.time()
        whole = self.video.shape[1]*self.video.shape[2]   
        
        skipped_corr = 0
        skipped_peak = 0
        for x in range(self.video.shape[2]):
            for y in range(self.video.shape[1]):
                
                if np.abs(self.video[:, y, x]).max() > noise_level:
                    corr_out = ip.correlation_temporal(self.video[:, y, x], self.k_diff, dip, threshold)
                    self.frames_binding[x][y] = corr_out['bind'][0] 
                    self.frames_unbinding[x][y] = corr_out['unbind'][0]
         
                    
                    if len(corr_out['bind'][0])!=0:
                        for f in corr_out['bind'][0]:
                            self.candidate.add((f, y, x))
                            self.intensity_binding.append(corr_out['bind'][1]) 
                            self.intensity_unbinding.append(corr_out['unbind'][1])  
                    else:
                        skipped_peak+=1
                else:
                    skipped_corr+=1
                    
                i+=1
                print('\r\t{}/ {}, remains {:.2f} s'.format(i+1, whole, (tt.time()-time)/i*(whole-i)), end="") 
        print(' DONE')
        print('#PXS excluded from correlation: {} / {}, {:.1f} %'.format(skipped_corr, whole, skipped_corr/whole*100))
        print('#PXS excluded from peaks: {} / {}, {:.1f} %'.format(skipped_peak, whole-skipped_corr, skipped_peak/(whole-skipped_corr)*100))
        
        self.show_pixels = True
        self.show_detected = True
        
        self.mask = self.process_mask()
        self.np_marks_positions = [[[],[]] for i in range(self.video.shape[0])]
        self.np_positions = [[[],[]] for i in range(self.video.shape[0])]
        
        print('Connecting detected pxs into patterns.', end="")
        while len(self.candidate) != 0:
            out = self.detect_nps(self.candidate.pop())
            
            self.candidate.difference_update(out)
            self.np_amount+=1
            
        print(' DONE')    
        
        print('Amount of detected binding events: {}'.format(self.np_amount))
        
    def characterize_nps(self):
        for npl in self.np_detected_info:
             f, y, x = npl[1]   
             
             print(y, x)
             ry = int(np.heaviside(y - 25, 1)*(y - 25))
             rx = int(np.heaviside(x - 25, 1)*(x - 25))
             print(ry, rx)
             raw = self.video[f, ry:y + 25, rx: x + 25]
             mask = np.full((raw.shape), False, dtype=bool)
             

             px_y = [y]*2
             px_x = [x]*2
             extreme_pxs = [px_y, px_x]
             
             for px in npl[0]:
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
#                 my = px[1] - y + ry
#                 mx = px[2] - x + rx
                 mask[my, mx] = True
                 
             dy = extreme_pxs[0][1]-extreme_pxs[0][0]+1           
             dx = extreme_pxs[1][1]-extreme_pxs[1][0]+1
                         
             measures = measure_new(raw, mask, [dx, dy])
             visualize_and_save(raw, measures, self.folder, self.file)
             
            
        
    def plot_np_amount(self):
        data_frame = []
        for npp in self.np_positions:
            data_frame.append(len(npp[1]))
        
        data_integral = [sum(data_frame[:i+1]) for i in range(len(data_frame))]
        
        fig_np, np_plot = plt.subplots()
        np_plot.grid(linestyle='--')
        np_plot.set_title('Count of binding events')
        np_plot.set_xlabel('time [min]')
        np_plot.set_ylabel('NP count [a. u.]')
        np_plot.yaxis.set_major_locator(MaxNLocator(integer=True))
        np_integral_plot = np_plot.twinx()
        
        self.make_frame_stats()
        np_plot.plot(data_integral, linewidth=1, color=blue, label='count in frame')
        
        np_integral_plot.plot(data_frame, linewidth=1, color=red, label='integral count', zorder = -1)

        fig_np.legend(loc=3)
        return fig_np, np_plot
        
    def np_pixels(self, inten_a=1e-04, inten_b=5e-4):
        """ 
        Need to rewrite for changed self.video handling
        """

        mask = np.zeros(self._video.shape[:2])
        for i in range(self._video.shape[0]):
            print('done {}/{}'.format(i, self._video.shape[0]))
            for j in range(self._video.shape[1]):
                try:
                    mask[i, j] = tl.t2i(is_np(self._video[i, j, :], inten_a, inten_b))
                except:
                    mask[i, j] = 0
                    print('no fit')
        return mask

    def np_count(self, mask, s1=2, s2=25, show=False):
        """ 
        Need to rewrite for changed self.video handling
        """
        
        gray = mask.astype(np.uint8)
        th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        s1 = 2
        s2 = 25
        xcnts = []
        control_mask = np.zeros(mask.shape)
        for cnt in cnts:
            if s1 < cv2.contourArea(cnt) < s2:

                for c in cnt:
                    control_mask[c[0][1], c[0][0]] = 1
                xcnts.append(cnt)
        if show:
            print("Dots number: {}".format(len(xcnts)))
            video_dark = (self.video[:, :, -1] + 1e-02) * 1e04
            video_rgb = cv2.cvtColor(video_dark.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            video_rgb[:, :, 0] += control_mask.astype(np.uint8) * 100
            plt.imshow(video_rgb)

        return len(xcnts)

    def np_AR(self):
        """ 
        Need to rewrite for changed self.video handling
        """
        
        self.np_count(self.np_pixels(), show=True)

    def explore(self, source='vid'):
        

        def frame_info(i):
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
#                is_np(self.video[:, y, x], show=True)
                ip.correlation_temporal(ax.volume[:, y, x], k_diff=self.k_diff, step=self.dip, threshold=self.threshold,  show=True)

        def next_slice(i):
            ax.index = (ax.index + i) % ax.volume.shape[0]
            img.set_array(ax.volume[ax.index])
            
            if self.show_pixels:
                mask.set_array(volume_mask[ax.index])  
                
            if self.show_detected:
                [p.remove() for p in reversed(ax.patches)]
                if self._img_type == 'diff' or self._img_type == True:
                    for i in range(len(self.np_marks_positions[ax.index][1])):
                        p = mpatches.Circle(
                                (self.np_marks_positions[ax.index][0][i], self.np_marks_positions[ax.index][1][i]), 
                                5, 
                                color=red, 
                                fill = False, 
                                lw = 2)
                        ax.add_patch(p)
                elif self._img_type == 'int' or self._img_type == False:
                    for npp in self.np_marks_positions[:ax.index]:
                        for i in range(len(npp[1])):
                            p = mpatches.Circle(
                                    (npp[0][i], npp[1][i]), 
                                    5, 
                                    color=red, 
                                    fill = False, 
                                    lw = 1)
                            ax.add_patch(p)
                    
            if self.show_stats:
                location.xy=[ax.index, -1]
                fig_stat.canvas.draw()            
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
                    for npp in self.np_marks_positions[:ax.index]:
                        for i in range(len(npp[1])):
                            p = mpatches.Circle(
                                    (npp[0][i], npp[1][i]), 
                                    5, 
                                    color=red, 
                                    fill = False, 
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
            stat_plot.set_title('STD of each frame')
            stat_plot.set_xlabel('time [min]')
            stat_plot.set_ylabel('intensity [a. u.]')
            
            if self.np_amount > 0:
                np_plot = stat_plot.twinx()
            
            self.make_frame_stats()
            stat_plot.plot(self.stats_std, linewidth=1, color=yellow, label='stdev')
            stat_plot.plot([np.average(self.stats_std) for i in self.stats_std], linewidth=1, color=blue, label='average', ls=':')  
            stat_plot.set_ylim((0, 0.003))
            
            if self.np_amount > 0:
                data_frame = []
                for npp in self.np_positions:
                    data_frame.append(len(npp[1]))
            
                data_integral = [sum(data_frame[:i+1]) for i in range(len(data_frame))]
            
               
                np_plot.yaxis.set_major_locator(MaxNLocator(integer=True))
                np_plot.plot(data_frame, linewidth=1, color=black, label='integral count', ls=':')  
                np_plot.plot(data_integral, linewidth=1, color=black, label='count in frame')          
            
            location = mpatches.Rectangle((ax.index, -1), 1/60, 5, color=red)                
            stat_plot.add_patch(location)
            fig_stat.legend(loc=3)
                        
                        
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
