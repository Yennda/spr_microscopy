import numpy as np
from video_processing import Video
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
#from matplotlib.backend_bases import LocationEvent

from PIL import Image
import tkinter as tk
from np_analysis import np_analysis, is_np
from classes import Cursor

FOLDER_NAME = '/exports'
NAME_LOCAL_SPR = 'spr'
NAME_GLOBAL_SPR = 'spr_integral'


yellow='#ffb200'
red='#DD5544'
blue='#0284C0'
black='#000000'

COLORS=[yellow, blue, red, black]


class BioVideo():
    def __init__(self, folder, file, channels):
        self.folder = folder
        self.file = file
        self._channels = [c for c in range(channels)]
        self._videos = None
        self.time_info = None
        self.rng = None
        

        self.spr_time = None
        self.spr_signals = None
        self.syn_index = None
        
        self.spr = False
        self.ref_frame = 0
        self._img_type = 'raw'
        self.orientation = None #True-horizontal, False-vertical
        
    def loadData(self):
        self._videos = []
        print('Don\'t forget to run one of the "make_..." methods ')
        for c in self._channels:
            video = Video(self.folder, self.file+'_{}'.format(c+1))
            video.loadData()
            video.rng = [0, 1]
            self._videos.append(video)
        self.orientation = self._videos[0].video_stats[1][1] < self._videos[0].video_stats[1][0]
                        
        self.time_info=self._videos[0].time_info
        self.rng=self._videos[0].rng
        
        self.loadSPR()
        self.synchronization()
        
    def loadSPR(self):
        self.spr_signals=[]
        
        for c in self._channels:
            f= open(self.folder+NAME_GLOBAL_SPR+'_{}.tsv'.format(c+1), 'r')
            contents=f.readlines()
        
            time=[]
            signal=[]
    
            for line in contents[:-1]:
                line_split=line.split('\t')
                
                if c == 0:
                    time.append(float(line_split[0]))
                signal.append(float(line_split[1])) 
                
            if c == 0:
                self.spr_time = time
            self.spr_signals.append(signal)
            
    def fouriere(self):
        for video in self._videos:
            video.fouriere()
        
    def make_diff(self):
        for video in self._videos:
            video.make_diff()
        self.rng = [-0.01, 0.01]
        self._img_type = 'diff'
            
    def make_int(self):
        for video in self._videos:

            video.make_int()
        self.rng = [-0.01, 0.01]
        self._img_type = 'int'
        
    def make_toggle(self):
        for video in self._videos:
            video.make_diff()
            video.make_int()
            video._img_type = True
        self.rng = [-0.01, 0.01]
        self._img_type = 'toggle'
            
    def make_both(self):
        for video in self._videos:
            video.make_diff()
            video.make_int()
            video._img_type = True
        self.rng = [-0.01, 0.01]
        self._img_type = 'both'
        self._channels = [i for i in range(len(self._channels)*2)]
        
        
    def synchronization(self):
        
        f= open(self.folder+NAME_LOCAL_SPR+self.file[3:]+'_{}.tsv'.format(self._channels[0]+1), 'r')
        contents=f.readlines()
        
        signal=[]
        for line in contents[:2]:
            signal.append(float(line.split('\t')[1])) 
                
        for i in range(len(self.spr_time)):
            if self.spr_signals[0][i:i+2]==signal:
                self.syn_index = i
                break
            elif i==len(self.spr_time)-2:
                raise Exception('Could not match global and local SPR signals.')
                
    def save_frame(self, channel, frame):
        """
        Saves the specified frame of the specified channel as a tiff file
        
        Parameters:
            channel (int): number of the channel e. g. 1, 2, ...
            
        Returns:
            no return
            
        """
        channel=int(channel)
        frame=int(frame)
        video = self._videos[channel-1].video
        name = '{}/{}_{}-{}'.format(self.folder+FOLDER_NAME, self._img_type,
                                                              video.shape[0],
                                                              frame)
        
        image_bw = (video[:,:, frame]+np.ones(video.shape[:2])*self.rng[1])*256*50
        image = Image.fromarray(image_bw)
        image_rgb = image.convert('RGB')
        image_rgb.save(name + '.png')
        print('File SAVED @{}'.format(name))
        
    def save_array(self, channel, start, end):
        for i in range(int(start.get()), int(end.get())):
            self.save_frame(int(channel.get()), i)
        
              
    def save_array_form(self):
        master = tk.Tk()
        tk.Label(master, text="channel").grid(row=0)
        tk.Label(master, text="start").grid(row=1)
        tk.Label(master, text="end").grid(row=2)
        
        channel = tk.Entry(master)
        start = tk.Entry(master)
        end = tk.Entry(master)
        
        channel.grid(row=0, column=1)
        start.grid(row=1, column=1)
        end.grid(row=2, column=1)
        
        
        tk.Button(master, 
                    text='Save', command=(lambda start = start, end = end, channel = channel: self.save_array(channel, start, end))).grid(row=3, 
                                                       column=1, 
                                                       sticky=tk.W, 
                                                       pady=4)
        
        master.mainloop()

    def explore(self, show='all'):
        def frame_info(c, i):
            return '{}/{}  t= {:.1f} min dt= {:.2f} s'.format(
                i,
                axes[c].volume.shape[0],
                self.spr_time[self.syn_index+i],
                self.time_info[i][1]
            )

        def mouse_scroll(event):
            fig = event.canvas.figure
            axes = fig.axes
            if event.button == 'down':
                next_slice(1)
            elif event.button == 'up':
                next_slice(-1)
            fig.canvas.draw()

        def mouse_click(event):
            pass
            if event.button == 3:
                self.np_number+=1
                print(self.np_number)
                fig = event.canvas.figure
                ax = fig.axes[0]
                x = int(event.xdata)
                y = int(event.ydata)
                raw = volume[axes[1].index]
                np_analysis(raw[y - 25: y + 25, x - 25:x + 25], self.folder, self.file)

                p = mpatches.Rectangle((x - 0.5, y - 0.5), 5, 5, color='#FF0000', alpha=0.5)
                axes[1].add_patch(p)
                print('you pressed', event.button, event.xdata, event.ydata)
                fig.canvas.draw()
            elif event.dblclick:
                x = int((event.xdata - 0.5) // 1)
                y = int((event.ydata - 0.5) // 1)
                #                file = open('data.txt', 'a')
                #                file.write('['+', '.join([str(i) for i in self._video[y, x,:]])+'],\n')
                #                file.close()

                print(is_np(self._video[y, x, :], show=True))

        # Next slice func.
        def next_slice(i):
            volume_list = [axes[c].volume for c in self._channels]
            
            for c in self._channels:
                axes[c].index = (axes[c].index + i) % volume_list[c].shape[0]
                img[c].set_array(volume_list[c][axes[c].index])
            fig.suptitle(frame_info(c, axes[c].index))
            
            if self.spr:
                location.xy=[self.spr_time[self.syn_index+axes[0].index], -1]
         
        def button_press(event):
            fig = event.canvas.figure
            if event.key == '6':
                fig = event.canvas.figure
                next_slice(10)
                fig.canvas.draw()
            elif event.key == '4':
                fig = event.canvas.figure
                next_slice(-10)
                fig.canvas.draw()
#            elif event.key == 'x':
#                [p.remove() for p in reversed(axes[1].patches)]
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
            elif event.key == '5':
                self.rng = [i * 1.2 for i in self.rng]
                for im in img:
                    im.set_clim(self.rng)
            elif event.key == '8':
                self.rng = [i / 1.2 for i in self.rng]
                for im in img:
                    im.set_clim(self.rng)
            elif event.key == 't':
                if self._img_type != 'toggle':
                    return
                
                fig = event.canvas.figure
                for c in self._channels:
                    axes[c].volume = self._videos[c].video
                    
#                    if axes[c].int:
#                        axes[c].volume = np.swapaxes(np.swapaxes(self._videos[c]._video_diff, 0, 2), 1, 2)
#                        axes[c].int = False
#                    else:
#                        axes[c].volume = np.swapaxes(np.swapaxes(self._videos[c]._video_int, 0, 2), 1, 2)
#                        axes[c].int = True
                next_slice(0)
                fig.canvas.draw()
                
            elif event.key == 'd':
                pass
                 
            elif event.key == 'a':
                # checks and eventually creates the folder 'export_image' in the folder of data
                if not os.path.isdir(self.folder + FOLDER_NAME):
                    os.mkdir(self.folder + FOLDER_NAME)

                # creates the name, appends the rigth numeb at the end

                name = '{}/{}_T{:03.0f}_dt{:03.0f}'.format(self.folder+FOLDER_NAME, self.file,
                                                                      self.time_info[axes[1].index][0],
                                                                      self.time_info[axes[1].index][1] * 100)

                i = 1
                while os.path.isfile(name + '_{:02d}.png'.format(i)):
                    i += 1
                name += '_{:02d}'.format(i)

                # saves the png file of the view

                fig.savefig(name + '.png', dpi=300)

                xlim = [int(i) for i in axes[1].get_xlim()]
                ylim = [int(i) for i in axes[1].get_ylim()]

#                # saves the exact nad precise tiff file
#                pilimage = Image.fromarray(img.get_array()[ylim[1]:ylim[0], xlim[0]:xlim[1]])
#                pilimage.save(name + '.tiff')
#                print('File SAVED @{}'.format(name))

#            img.set_array(volume[axes[1].index])
            fig.canvas.draw_idle()

        
        if self.orientation:
            fig, axes = plt.subplots(nrows=len(self._channels)+int(self.spr), ncols=1)
        else:
            fig, axes = plt.subplots(ncols=len(self._channels)+int(self.spr), nrows=1)
        
        if self._channels == [0] and not self.spr:
            axes=[axes]
#        else:
#            if self.orientation:
#                fig, axes_all = plt.subplots(nrows=len(self._channels)+1, ncols=1)
#            else:
#                fig, axes_all = plt.subplots(ncols=len(self._channels)+1, nrows=1)
        if self.spr:
            spr_plot = axes[0]
            axes = axes[1:]
            
            spr_plot.grid(linestyle='--')
            spr_plot.set_title('SPR signal')
            spr_plot.set_xlabel('time [min]')
            spr_plot.set_ylabel('intensity [a. u.]')
            
            if self._img_type=='both':
                channels = [i for i in range(len(self._channels)//2)]
            else:
                channels = self._channels
                
            for c in channels: 
                spr_plot.plot(self.spr_time, self.spr_signals[c], linewidth=1, color=COLORS[c], label='ch. {}'.format(c+1))
                
            location = mpatches.Rectangle((self.spr_time[self.syn_index], -1), 1/60, 5, color=red)                
            spr_plot.add_patch(location)
            spr_plot.legend(loc=3)
                
        
        img=[]

        channel_type =  ['' ,' \ndifferential']
        for c in self._channels:
                
            if self._img_type == 'both':
                axes[c].volume = self._videos[c//2].video
                if self.orientation:
                    axes[c].set_ylabel('channel {}.{}'.format(c//2+1, channel_type[c%2]))
                else:
                    axes[c].set_xlabel('channel {}.{}'.format(c//2+1, channel_type[c%2]))
            else:
                axes[c].volume = self._videos[c].video
                if self.orientation:
                    axes[c].set_xlabel('channel {}.'.format(c+1))
                else:
                    axes[c].set_xlabel('channel {}.'.format(c+1)) 

            axes[c].index = 0

            img.append(axes[c].imshow(axes[c].volume[axes[c].index], cmap='gray', vmin=self.rng[0], vmax=self.rng[1]))
            
            fontprops = fm.FontProperties(size=10)
            scalebar = AnchoredSizeBar(axes[c].transData,
                   34, '100 $\mu m$', 'lower right', 
                   pad=0.1,
                   color='black',
                   frameon=False,
                   size_vertical=1,
                   fontproperties=fontprops)
            axes[c].add_artist(scalebar)
            
                   
#        cursor = Cursor(axes[0])
        
        fig.canvas.mpl_connect('scroll_event', mouse_scroll)
        fig.canvas.mpl_connect('button_press_event', mouse_click)
        fig.canvas.mpl_connect('key_press_event', button_press)    
#        fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)
        
        fig.suptitle(frame_info(c, axes[c].index))

#        fig.colorbar(img[0], ax=axes.ravel().tolist())
        
        
        plt.tight_layout()
        

        plt.show()

        print('''
-------------------------------------------------------------------------------
Basic shortcuts 

"8"/"5" increases/decreases contrast
Mouse scrolling moves the time 
"1" and "3" jumps 1 frames in time
"4" and "6" jumps 10 frames in time
"7" and "9" jumps 100 frames in time
"f" fulscreen
"o" zooms chosen area
"s" saves the figure
"t" toggles differential and integral image, when the method "make_both" is used

Official MATPLOTLIB shortcuts at https://matplotlib.org/users/navigation_toolbar.html
-------------------------------------------------------------------------------
              ''')