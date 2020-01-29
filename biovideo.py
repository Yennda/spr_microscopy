import numpy as np
from video_processing import Video
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.backend_bases import LocationEvent

from np_analysis import np_analysis, is_np
import tools as tl

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
        
        self.spr = False
        self.spr_time = None
        self.spr_signals = None
        self.syn_index = None
        self.ref_frame = 0
        
    def loadData(self):
        self._videos = []
        print('Don\'t forget to run a method "make_int" or "make_diff". ')
        for c in self._channels:
            video = Video(self.folder, self.file+'_{}'.format(c+1))
            video.loadData()
            video.rng = [-0.01, 0.01]
            
            video._video=video._video[:,200:1000,:2]
            video.refresh()
            video.make_int()
            
            self._videos.append(video)
            
            
            
            # odtud je to jen bastleni kodu
           # video2 = Video(self.folder, self.file+'_{}'.format(c+1))
            #video2.loadData()
            #video2.rng = [-0.01, 0.01]
            
           # video2._video=video2._video[:,200:1000,:]
            #video2.refresh()
           # video2.make_diff()
            
            #self._videos.append(video2)            
        
        #self._channels = [c for c in range(2*len(self._channels))]
        #dotud
            
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
            
    def makediff(self):
        for video in self._videos:
            video.make_diff()
            video.refresh()
    def makeint(self):
        for video in self._videos:
            video.ref_frame = self.ref_frame
            video.make_int()
            video.refresh()           
            
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

    def explore(self, show='all'):
        def frame_info(c, i):
            return '{}/{}  t= {:.1f} s dt= {:.2f} s'.format(
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
            elif event.key == '5':
                self.rng = [i * 1.2 for i in self.rng]
                for im in img:
                    im.set_clim(self.rng)
            elif event.key == '8':
                self.rng = [i / 1.2 for i in self.rng]
                for im in img:
                    im.set_clim(self.rng)
#            elif event.key == 'p':
#                self.np_number=0
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



        

        
        
        if not self.spr:
            fig, axes = plt.subplots(nrows=len(self._channels), ncols=1)
            
        else:
            fig, axes_all = plt.subplots(nrows=len(self._channels)+1, ncols=1)
            spr_plot = axes_all[0]
            axes = axes_all[1:]
            
            spr_plot.grid(linestyle='--')
            spr_plot.set_title('SPR signal')
            spr_plot.set_xlabel('time [min]')
            spr_plot.set_ylabel('intensity [a. u.]')
            
            for c in self._channels:            
                spr_plot.plot(self.spr_time, self.spr_signals[c], linewidth=1, color=COLORS[c], label='ch. {}'.format(c+1))
                
            location = mpatches.Rectangle((self.spr_time[self.syn_index], -1), 1/60, 5, color=red)       

            
            spr_plot.add_patch(location)

            spr_plot.legend(loc=3)
        
        
        
        img=[]
        
        for c in self._channels:
            axes[c].volume = np.swapaxes(np.swapaxes(self._videos[c]._video_new, 0, 2), 1, 2)
            axes[c].index = 0
            axes[c].set_ylabel('channel {}.'.format(c+1))    
#            axes[c].spines[].set_color(COLORS[c])

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
        fig.canvas.mpl_connect('scroll_event', mouse_scroll)
        fig.canvas.mpl_connect('button_press_event', mouse_click)
        fig.canvas.mpl_connect('key_press_event', button_press)    
        
        fig.suptitle(frame_info(c, axes[c].index))

#        fig.colorbar(img[0], ax=axes.ravel().tolist())
        
        
#        plt.tight_layout()
        plt.show()

        print('''
-------------------------------------------------------------------------------
Basic shortcuts 

"8"/"5" increases/decreases contrast
Mouse scrolling moves the time 
"4" and "6" jumps 10 frames in time
"f" fulscreen
"o" zooms chosen area
"s" saves the figure

Official MATPLOTLIB shortcuts at https://matplotlib.org/users/navigation_toolbar.html
-------------------------------------------------------------------------------
              ''')