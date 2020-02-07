import numpy as np
from video_processing import Video
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
#from matplotlib.backend_bases import LocationEvent

from PIL import Image
from np_analysis import np_analysis, is_np
from classes import Cursor

FOLDER_NAME = '/exports_bio'
NAME_LOCAL_SPR = 'spr'
NAME_GLOBAL_SPR = 'spr_integral'


yellow='#ffb200'
red='#DD5544'
blue='#0284C0'
black='#000000'

COLORS = [yellow, blue, red, black]
SIDES = ['left', 'right', 'bottom', 'top']


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
        self._toggle = True
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
            video.ref_frame = self.ref_frame
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
#            axes = fig.axes
            if event.button == 'down':
                next_slice(1)
            elif event.button == 'up':
                next_slice(-1)
            fig.canvas.draw()

        def mouse_click(event):
            if event.button == 3:
                axes_chosen = event.inaxes
                
                channel = axes.index(axes_chosen)+1
                
                if self.orientation:
                    img_type = axes_chosen.get_ylabel()
                else:
                    img_type = axes_chosen.get_xlabel()
                
                if not os.path.isdir(self.folder + FOLDER_NAME):
                    os.mkdir(self.folder + FOLDER_NAME)
                    
                name = '{}/{}_frame_{}-{}'.format( 
                        self.folder+FOLDER_NAME,
                        img_type.replace(' ', '_'),
                        axes_chosen.index,
                        axes_chosen.volume.shape[0])
                i = 1
                while os.path.isfile(name + '_{:02d}.png'.format(i)):
                    i += 1
                    
                name += '_{:02d}'.format(i)

                xlims = [int(i+0.5) for i in list(event.inaxes.get_xlim())]
                ylims = [int(i+0.5) for i in list(event.inaxes.get_ylim())]
                image_load = axes_chosen.volume[axes_chosen.index][ylims[1]:ylims[0], xlims[0]:xlims[1]]
                image_bw = (image_load-np.ones(image_load.shape[:2])*self.rng[0])*256*(self.rng[1]-self.rng[0])**-1


                image = Image.fromarray(image_bw)
                image_rgb = image.convert('RGB')
                image_rgb.save(name + '.png')
                print('File SAVED @ {}.png'.format(name.replace('{}/'.format(self.folder), '')))

        # Next slice func.
        def next_slice(i):
            volume_list = [axes[c].volume for c in self._channels]
            
            for c in self._channels:
                axes[c].index = (axes[c].index + i) % volume_list[c].shape[0]
                img[c].set_array(volume_list[c][axes[c].index])
            fig.suptitle(frame_info(c, axes[c].index))
            
            if self.spr:
                location.xy=[self.spr_time[self.syn_index+axes[0].index], -1]
                fig_spr.canvas.draw()
                fig_spr.suptitle(frame_info(c, axes[c].index))
         
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
                    
                    if self._toggle:
                        self._toggle = False
                        if self.orientation:
                            axes[c].set_ylabel('diff {}'.format(c+1))
                        else:
                            axes[c].set_xlabel('diff {}'.format(c+1)) 
                    else:
                        self._toggle = True
                        if self.orientation:
                            axes[c].set_ylabel('int {}'.format(c+1))
                        else:
                            axes[c].set_xlabel('int {}'.format(c+1)) 
                    
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

#                xlim = [int(i) for i in axes[1].get_xlim()]
#                ylim = [int(i) for i in axes[1].get_ylim()]

#                # saves the exact nad precise tiff file
#                pilimage = Image.fromarray(img.get_array()[ylim[1]:ylim[0], xlim[0]:xlim[1]])
#                pilimage.save(name + '.tiff')
#                print('File SAVED @{}'.format(name))

#            img.set_array(volume[axes[1].index])
            fig.canvas.draw_idle()

        
        if self.orientation:
            fig, axes = plt.subplots(nrows=len(self._channels), ncols=1)
        else:
            fig, axes = plt.subplots(ncols=len(self._channels), nrows=1)
        
        if len(self._channels) == 1:
            axes=[axes]
        else:
            axes = list(axes)
        
        img=[]

        channel_type =  ['int' ,' diff']
        
        
        for c in self._channels:
                
            if self._img_type == 'both':
                axes[c].volume = self._videos[c//2].video
                if self.orientation:
                    axes[c].set_ylabel('{} {}'.format(channel_type[c%2], c//2+1))
                else:
                    axes[c].set_xlabel('{} {}'.format(channel_type[c%2], c//2+1))
                for s in SIDES:
                    axes[c].spines[s].set_color(COLORS[c%2])
            elif self._img_type == 'toggle':
                axes[c].volume = self._videos[c].video
                if self.orientation:
                    axes[c].set_ylabel('int {}'.format(c+1))
                else:
                    axes[c].set_xlabel('int {}'.format(c+1)) 
                for s in SIDES:
                    axes[c].spines[s].set_color(COLORS[c])
            else:
                axes[c].volume = self._videos[c].video
                if self.orientation:
                    axes[c].set_ylabel('{} {}'.format(self._img_type, c+1))
                else:
                    axes[c].set_xlabel('{} {}'.format(self._img_type, c+1)) 
                for s in SIDES:
                    axes[c].spines[s].set_color(COLORS[c])
            
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
        if self.spr:
            fig_spr, spr_plot = plt.subplots()
        
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
            
            fig_spr.suptitle(frame_info(c, axes[c].index))
            
        
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