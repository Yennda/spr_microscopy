import numpy as np
from video_processing import Video
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from np_analysis import np_analysis, is_np
import tools as tl

FOLDER_NAME = '/exports'

class BioVideo(Video):
    def __init__(self, folder, file):
        super().__init__(folder, file)
        self._channels=2
        
    def explore(self, source='vid'):
        
        if not self.show_original:
            data = np.swapaxes(np.swapaxes(self._video_new, 0, 2), 1, 2)
        else:
            data = np.swapaxes(np.swapaxes(self.video, 0, 2), 1, 2)

        def frame_info(i):
            return '{}/{}  t= {} s dt= {:.2f} s'.format(
                i,
                volume.shape[0],
                tl.SecToMin(self.time_info[i][0]),
                self.time_info[i][1]
            )

        def mouse_scroll(event):
            fig = event.canvas.figure
            axes = fig.axes
            if event.button == 'down':
                next_slice(axes, 1)
            elif event.button == 'up':
                next_slice(axes, -1)
            fig.canvas.draw()

        def mouse_click(event):
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
        def next_slice(axes, i):
            volume = [axes[i].volume for i in range(self._channels)]
            
            for j in range(self._channels):
                axes[j].index = (axes[j].index + i) % volume[j].shape[0]
                img.set_array(volume[j][axes[j].index])
                axes[j].set_title(frame_info(axes[j].index))

        def button_press(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            volume = data
            if event.key == 'right':
                fig = event.canvas.figure
                ax = fig.axes[0]
                next_slice(ax, 10)
                fig.canvas.draw()
            elif event.key == 'left':
                fig = event.canvas.figure
                ax = fig.axes[0]
                next_slice(ax, -10)
                fig.canvas.draw()
            elif event.key == 'x':
                [p.remove() for p in reversed(axes[1].patches)]
            elif event.key == 'm':
                lim = [i * 1.2 for i in img.get_clim()]
                img.set_clim(lim)
            elif event.key == 'j':
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

                # saves the exact nad precise tiff file
                pilimage = Image.fromarray(img.get_array()[ylim[1]:ylim[0], xlim[0]:xlim[1]])
                pilimage.save(name + '.tiff')
                print('File SAVED @{}'.format(name))

            img.set_array(volume[axes[1].index])
            fig.canvas.draw_idle()


        

        fig, axes = plt.subplots(nrows=2, ncols=1)
        volume = data
        
        for i in range(self._channels):
            axes[i].volume = volume
            axes[i].index = 0
            axes[i].set_title('{}/{}  t= {:.2f} s dt= {:.2f} s'.format(axes[i].index, volume.shape[0], self.time_info[axes[i].index][0],
                                                                  self.time_info[axes[i].index][1]))
    
            if source == 'diff' or source == 'vid':
                img = axes[i].imshow(volume[axes[i].index], cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
            else:
                img = axes[i].imshow(volume[axes[i].index], cmap='gray')
            fig.canvas.mpl_connect('scroll_event', mouse_scroll)
            fig.canvas.mpl_connect('button_press_event', mouse_click)
            fig.canvas.mpl_connect('key_press_event', button_press)
    
            fontprops = fm.FontProperties(size=10)
            scalebar = AnchoredSizeBar(axes[i].transData,
                       34, '100 $\mu m$', 'lower right', 
                       pad=0.1,
                       color='black',
                       frameon=False,
                       size_vertical=1,
                       fontproperties=fontprops)
    
            axes[i].add_artist(scalebar)
        
#        cb = fig.colorbar(img, ax=ax)
        plt.tight_layout()
        plt.show()

        print('''
Buttons "j"/"m" serve to increasing/decreasing contrast 
Button "s" saves the current image as tiff file
Mouse scrolling moves to neighboring frames
Official shortcuts here https://matplotlib.org/users/navigation_toolbar.html
Right mouse button click selects and switches to analysis of chosen NP image
Double click plots the intensity course of the pixel and decides if it includes NP
              ''')