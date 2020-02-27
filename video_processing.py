import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import image_processing as ip
from multiprocessing import Pool
import time as t

from np_analysis import np_analysis, is_np
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
        self.np_positions = None
        
        self.threshold = None 
        self.show_graphic = True
        self.show_pixels = False
        self.show_detected = False

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
            
            
            print('\r{}/ {}'.format(i+1, self.video_stats[1][2]), end="")
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
        
        print('Computing the integral imag')
        
        for i in range(1, sh[-1]):
            print('\r{}/ {}'.format(i+1, self.video_stats[1][2]), end="")
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
        print('Computing the mask fro image')
        for x in range(self.video.shape[2]):
            for y in range(self.video.shape[1]):
                for f in self.frames_binding[x][y]:
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 1] = [1]*2*k_diff
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 3] = tri
                    
                for f in self.frames_unbinding[x][y]:
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 0] = [1]*2*k_diff
                    volume_mask[f-k_diff:(f+k_diff)%self.video.shape[0], y, x, 3] = tri
                i += 1
                print('\r{}/ {}'.format(i+1, imax), end="")
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
                print('\r{}/ {}'.format(i+1, imax), end="")
        print(' DONE')
        return volume_mask
        
    def make_diff(self, k = 1):
        self._video['diff'] = self.process_diff(k)
        self._img_type = 'diff'

    def make_int(self, k = 1):
        self._video['int']= self.process_int(k)
        self._img_type = 'int'
    
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

    def fouriere(self, show = False):
        print('Filtering fouriere frequencies')
        if type(self._img_type) == bool:
            img_type = ['int']
        else:
            img_type = [self._img_type]
        for it in img_type:
            for i in range(self._video[it].shape[2]):

                print('\r{}/ {}'.format(i+1, self.video_stats[1][2]), end="")    
                f = np.fft.fft2(self._video[it][:, :, i])
                
                magnitude_spectrum = 20 * np.log(np.abs(f))
                mask = np.real(magnitude_spectrum) > 30
                f[mask] = 0
                  
                img_back = np.fft.ifft2(f)
                self._video[it][:, :, i] = np.real(img_back)
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
            
            found_pxs = self.mask[f-2:f+3, y-1:y+2, x-1:x+2].nonzero()

            for i in range(len(found_pxs[0])):
                points_to_do.add((f+found_pxs[0][i]-1, y+found_pxs[1][i]-1, x+found_pxs[2][i]-1))
                
            points_done.add((f, y, x))
            
        npf=int(round(np.average([p[0] for p in points_done])))
        npy=np.average([p[1] for p in points_done])
        npx=np.average([p[2] for p in points_done])
        
        for i in range(-self.k_diff//2, self.k_diff//2):
            self.np_positions[npf+i][0].append(npx)    
            self.np_positions[npf+i][1].append(npy)          
        return points_done
        
    def img_process_alpha(self, threshold = 15):
        if self._img_type != 'diff':
            print('Processes only differential image. Use make_diff method first.')
            return
        
        print('Correlation')
        self.threshold = threshold
        self.frames_binding = [[[] for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]
        self.frames_unbinding = [[[] for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]
        self.intensity_binding = [[[] for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]
        self.intensity_unbinding = [[[] for y in range(self.video.shape[1])] for x in range(self.video.shape[2])]

        i = 0
        time = t.time()
        whole = self.video.shape[1]*self.video.shape[2]   
        for x in range(self.video.shape[2]):
            for y in range(self.video.shape[1]):
                
                if np.abs(self.video[:, y, x]).max() > 0.001:
                    out = ip.correlation_temporal(self.video[:, y, x], 10, -0.003, threshold)
                    self.frames_binding[x][y] = out[0] 
                    self.frames_unbinding[x][y] = out[1]
                    self.intensity_binding[x][y] = out[2] 
                    self.intensity_unbinding[x][y] = out[3]           
                    
                    if len(out[0])!=0:
                        for f in out[0]:
                            self.candidate.add((f, y, x))
                    
                i+=1
                print('\r{}/ {}, remains {:.2f} s'.format(i+1, whole, (t.time()-time)/i*(whole-i)), end="") 
        print(' DONE')
        self.show_pixels = True
        self.show_detected = True
        
        self.mask = self.process_mask()
        self.np_positions = [[[],[]] for i in range(self.video.shape[0])]
        
        print('Connecting detected pxs into patterns.', end="")
        while len(self.candidate) != 0:
            out = self.detect_nps(self.candidate.pop())
            self.candidate.difference_update(out)
            self.np_amount+=1
            
        print(' DONE')    
        
        print('Amount of detected binding events: {}'.format(self.np_amount))
        
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
                volume.shape[0],
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
                raw = volume[ax.index]
                np_analysis(raw[y - 25: y + 25, x - 25:x + 25], self.folder, self.file)

                p = mpatches.Rectangle((x - 0.5, y - 0.5), 5, 5, color='#FF0000', alpha=0.5)
                ax.add_patch(p)
                print('you pressed', event.button, event.xdata, event.ydata)
                fig.canvas.draw()
                
            elif event.dblclick:
                x = int((event.xdata + 0.5) // 1)
                y = int((event.ydata + 0.5) // 1)
                #                file = open('data.txt', 'a')
                #                file.write('['+', '.join([str(i) for i in self._video[y, x,:]])+'],\n')
                #                file.close()
                print('------------')
                print('x = {}'.format(x))
                print('y = {}'.format(y))
#                is_np(self.video[:, y, x], show=True)
                ip.correlation_temporal(self.video[:, y, x], 10, -0.003, threshold=self.threshold,  show=True)

        def next_slice(i):
            ax.index = (ax.index + i) % volume.shape[0]
            img.set_array(volume[ax.index])
            if self.show_pixels:
                mask.set_array(volume_mask[ax.index])  
            if self.show_detected:
                [p.remove() for p in reversed(ax.patches)]
#                ax.scatter(self.np_positions[ax.index][0], self.np_positions[ax.index][1], s=80, facecolors='none', edgecolors='r')
                for i in range(len(self.np_positions[ax.index][1])):
                    p = mpatches.Circle(
                            (self.np_positions[ax.index][0][i], self.np_positions[ax.index][1][i]), 
                            5, 
                            color=red, 
                            fill = False, 
                            lw = 2)
                    ax.add_patch(p)

            ax.set_title(frame_info(ax.index))

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
            elif event.key == 'm':
                
                if self.show_graphic:
                    img.set_zorder(10)
                    self.show_graphic = False
                else:
                    img.set_zorder(0)
                    self.show_graphic = True
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

                xlim = [int(i) for i in ax.get_xlim()]
                ylim = [int(i) for i in ax.get_ylim()]

                # saves the exact nad precise tiff file
                pilimage = Image.fromarray(img.get_array()[ylim[1]:ylim[0], xlim[0]:xlim[1]])
                pilimage.save(name + '.tiff')
                print('File SAVED @{}'.format(name))

            img.set_array(volume[ax.index])
            fig.canvas.draw_idle()


        fig, ax = plt.subplots()
        volume = self.video
        ax.volume = volume
        ax.index = 0
        ax.set_title('{}/{}  t= {:.2f} s dt= {:.2f} s'.format(ax.index, volume.shape[0], self.time_info[ax.index][0],
                                                              self.time_info[ax.index][1]))

        if self._img_type == 'raw':
            img = ax.imshow(volume[ax.index], cmap='gray')
        else:
            img = ax.imshow(volume[ax.index], cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
            
        if self.show_pixels:
            volume_mask = self.process_mask_image()
            ax.volume_mask = volume_mask
            mask = ax.imshow(volume_mask[ax.index])
            
        if self.show_detected:
            ax.scatter(self.np_positions[0][0], self.np_positions[0][1])
            
            
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
        
#        cb = fig.colorbar(img, ax=ax)
        plt.tight_layout()
        plt.show()
        print('='*50)
        print('''
BASIC SHORTCUTS

"8"/"5" increases/decreases contrast
Mouse scrolling moves the time 
"1" and "3" jumps 1 frames in time
"4" and "6" jumps 10 frames in time
"7" and "9" jumps 100 frames in time
"f" fulscreen
"o" zooms chosen area.
"a" saves the image
"s" saves the the whole figure
"m" disables the mask

"Left mouse button double click" show the time/intensity point of the pixel with the correlation function.

Official MATPLOTLIB shortcuts at https://matplotlib.org/users/navigation_toolbar.html
Buttons "j"/"m" serve to increasing/decreasing contrast 
Button "s" saves the current image as tiff file
Mouse scrolling moves to neighboring frames
Official shortcuts here https://matplotlib.org/users/navigation_toolbar.html
Right mouse button click selects and switches to analysis of chosen NP image
Double click plots the intensity course of the pixel and decides if it includes NP
              ''')
