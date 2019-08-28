import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math as m
import cv2

from PIL import Image
import os
  
from scipy.optimize import curve_fit

def SecToMin(sec):
    return '{:.0f}:{:.1f}'.format(sec//60, sec%60)

def h(x): return 0.5 * (np.sign(x) + 1)

def step(x, a, b0, b1): return (b1-b0) * (np.sign(x-a) + 1)+b0

def linear(x, a, b):
    return a*x + b

def find_step(data):
    return np.argmax([m.fabs(data[i]-data[i+2]) for i in range(len(data)-2)])

def t2i(boo):
    if boo:
        return 1
    else:
        return 0

def is_np(data, inten_a=1e-04, inten_b=5e-4, show=False):

    xdata=np.arange(len(data))
    popt_guess, pcov_guess = curve_fit(step, xdata, data, p0=[find_step(data),0, -5e-04], epsfcn=0.1)
    
    popt_fixed, pcov_fixed = curve_fit(step, xdata, data, p0=[int(len(data)/2),0, -5e-04], epsfcn=0.1)

    squares_guess=sum([(step(i, *popt_guess)-data[i])**2 for i in xdata])
    squares_fixed=sum([(step(i, *popt_fixed)-data[i])**2 for i in xdata]) 
    

    if squares_guess<squares_fixed:
        popt, pcov=popt_guess, pcov_guess
        squares=squares_guess
    else:
        popt, pcov=popt_fixed, pcov_fixed
        squares=squares_fixed   
    
    lpopt, lpcov = curve_fit(linear, xdata, data, p0=[1e-4, 0], epsfcn=0.1)
    lsquares=sum([(linear(i, *lpopt)-data[i])**2 for i in xdata])  
    
    
    if show:
#        print('a, b: {}'.format(popt))
        #    print(pcov)
        print('delta: {}'.format(m.fabs(popt[2]-popt[1])))
        print('step: {}'.format(squares))
        print('linear {}: '.format(lsquares))
        print(2*squares<lsquares)
#        print('variance: {}'.format(np.var(data)))

        
        fix, axes = plt.subplots()
        axes.plot(data,'b-', label='data')
        axes.plot(xdata, step(xdata, *popt), 'r-')  
        axes.plot(xdata, linear(xdata, *lpopt), 'g-')  
        
    return (m.fabs(popt[2]-popt[1])>inten_a and 2*squares<lsquares) or (m.fabs(popt[2]-popt[1])>inten_b and squares<lsquares) #or (np.abs(data[-1])>mx)

def frame_times(file_content):
    time0=int(file_content[1].split()[0])
    time_info=[]
    time_last=time0
    for line in file_content[1:]:
        time_actual=int(line.split()[0])
        time_info.append([(time_actual-time0)/1e7, (time_actual-time_last)/1e7])
        time_last=time_actual
    return time_info

class VideoRec(object):

    def __init__(self, folder, file):
        self.folder=folder
        self.file=file
        self.file_name = folder+file
        self.video_stats = None
        self._video = None
        self.view = None
        self.rng= [-1, 1]
        self.time_info=None

    def __iter__(self):
        self.n = -1
        self.MAX = self.video.shape[2]-1
        return self

    def __next__(self):
        if self.n < self.MAX:
            self.n += 1
            return self.video[:, :, self.n]
        else:
            raise StopIteration

    def loadData(self):
        self.video_stats = self.loadBinVideoStats()
        self._video = self.loadBinVideo()
        
    @property
    def video(self):
        return self._video[self.view[1]:self.view[1]+self.view[3], self.view[0]:self.view[0]+self.view[2],: ]

    def loadBinVideoStats(self):
        suffix = '.tsv'
        with open(self.file_name+suffix, mode='r') as fid:
            file_content = fid.readlines()
            
        self.time_info=frame_times(file_content)
        stats = file_content[1].split()
        video_length = len(file_content) - 1
        video_width = int(stats[1])
        video_hight = int(stats[2])
        video_fps = float(stats[4])*int(stats[5])
        self.view = [0, 0, video_width, video_hight]
        return [video_fps, [video_width, video_hight, video_length]]

    def loadBinVideo(self):
        code_format = np.float64  # type double
        suffix = '.bin'
        with open(self.file_name+suffix, mode='rb') as fid:
            video = np.fromfile(fid, dtype=code_format)
        video = np.reshape(video, (self.video_stats[1][0],
                                   self.video_stats[1][1],
                                   self.video_stats[1][2]), order='F')


        return np.swapaxes(video, 0, 1)
    
    def rescale(self, percent=50):
        self._video.shape
        width = int(self._video.shape[1] * percent/ 100)
        height = int(self._video.shape[0] * percent/ 100)
        dim = (width, height)
        dim_vid = ( height,width, self._video.shape[2])

        new=np.ndarray(dim_vid)
        
        for i in range(self._video.shape[2]):
           new[:,:,i]=cv2.resize(self._video[:,:,i], dim, interpolation =cv2.INTER_AREA)
        self._video=new

    def time_fouriere(self):
        middle=int(self._video.shape[2]/2)
        video=np.zeros(self._video.shape)
        for i in range(self._video.shape[0]):
            print('done {}/{}'.format(i, self._video.shape[0]))
            for j in range(self._video.shape[1]):
                signal=self._video[i, j,:]
                fspec=np.fft.fft(signal)
                
                fspec[middle-5:middle+5]=0
                
                
                video[i, j,:] = np.fft.ifft(fspec)
        self._video=video
        
        
    def fouriere(self):
        for i in range(self._video.shape[2]):
            f = np.fft.fft2(self.video[:,:,i])
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20*np.log(np.abs(fshift))
            
#            fshift[:, :530] = 0
#            fshift[:, 1120:] = 0
            #only strips, efficient
#            fshift[:, 490:550] = 0
#            fshift[:, 1120:1180] = 0
#            rows, cols = self._video[:,:,0].shape
#            crow,ccol = int(rows/2) , int(cols/2)
#            fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
#            fshift[111:136, 139:309] = 0
#            fshift[133:163, 1318:1522] = 0
            
            #pro K4 nejlepsi 30
            mask=np.real(magnitude_spectrum)>30
            fshift[mask]=0
            
            f_ishift = np.fft.ifftshift(fshift)
#            magnitude_spectrum_filtered = 20*np.log(np.abs(fshift))
            
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)
            self._video[:,:,i]=img_back
#            self._video[:,:,i]=cv2.blur(img_back,(2,2))
    
    def np_recognition(self, inten_a=1e-04, inten_b=5e-4,):
        mask=np.zeros(self._video.shape[:2])
        for i in range(self._video.shape[0]):
            print('done {}/{}'.format(i, self._video.shape[0]))
            for j in range(self._video.shape[1]):
                try:
                    mask[i,j]=t2i(is_np(self._video[i,j,:], inten_a, inten_b))
                except:
                    mask[i,j]=0
                    print('no fit')
        return mask
            
#   
    def frame(self, arg=1):

        if type(arg) == int:
            data = self.video[:, :, arg]
        else:
            data = np.array(arg)
        fig, ax = plt.subplots()
        ax.imshow(data, cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
        ax.set(xlabel='x [px]', ylabel='y [px]')
        return fig
    
    
    def explore(self, source='vid'):
        if source=='diff':
            data=np.swapaxes(np.swapaxes(self.video_diff,0,2),1,2) 
        elif source=='ref':
            data=np.swapaxes(np.swapaxes(self.video_ref,0,2),1,2) 
        else:
            data=np.swapaxes(np.swapaxes(self.video,0,2),1,2)
        #Mouse scroll event.
        
        def frame_info(i):
            return '{}/{}  t= {} s dt= {:.2f} s'.format(
                    i, 
                    volume.shape[0], 
                    SecToMin(self.time_info[i][0]), 
                    self.time_info[i][1]
                    )
            
        def mouse_scroll(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            if event.button == 'down':
                next_slice(ax, 1)
            elif event.button == 'up':
                next_slice(ax, -1)
            fig.canvas.draw()
            
        def mouse_click(event):
            if event.button==3:
                x=int((event.xdata-0.5)//1)
                y=int((event.ydata-0.5)//1)
#                file = open('data.txt', 'a')
#                file.write('['+', '.join([str(i) for i in self._video[y, x,:]])+'],\n')
#                file.close()
                
                print(is_np(self._video[y, x,:], show=True))
                
                
            else:
                print('wrong button')
#            
#            
#                fig = event.canvas.figure
#                ax = fig.axes[0]
#                x=int( event.xdata)
#                y=int( event.ydata)
#                raw=volume[ax.index]
#                np_analysis(raw[y-25: y+25, x-25:x+25], self.folder, self.file)
#                
#                p=mpatches.Rectangle((), 5, 5, color='#FF0000', alpha=0.5)
#                ax.add_patch(p)
#                print('you pressed', event.button, event.xdata, event.ydata)
#                fig.canvas.draw()  
                
        #Next slice func.
        def next_slice(ax, i):
            volume = ax.volume
            ax.index = (ax.index + i) % volume.shape[0]
            img.set_array(volume[ax.index])
            ax.set_title(frame_info(ax.index))
            

        def button_press(event):
            fig = event.canvas.figure
            ax = fig.axes[0]
            volume = data
            if event.key=='right':
                fig = event.canvas.figure
                ax = fig.axes[0]
                next_slice(ax, 10)
                fig.canvas.draw()
            elif event.key=='left':
                fig = event.canvas.figure
                ax = fig.axes[0]
                next_slice(ax, -10)
                fig.canvas.draw()
            elif event.key=='x':
                [p.remove() for p in reversed(ax.patches)]
            elif event.key=='m':
                lim=[i*1.2 for i in img.get_clim()]
                img.set_clim(lim)
            elif event.key=='j':
                lim=[i/1.2 for i in img.get_clim()]
                img.set_clim(lim)
            elif event.key=='a':
                #checks and eventually creates the folder 'export_image' in the folder of data
                if not os.path.isdir(self.folder+'/export_img'):
                    os.mkdir(self.folder+'/export_img')
                
                #creates the name, appends the rigth numeb at the end
                name='{}/export_img/{}_T{:03.0f}_dt{:03.0f}'.format(self.folder, self.file, self.time_info[ax.index][0], self.time_info[ax.index][1]*100)
                
                i=1
                while os.path.isfile(name+'_{:02d}.png'.format(i)):
                    i+=1
                name+='_{:02d}'.format(i)
                
                #saves the png file of the view
                
                fig.savefig(name+'.png' , dpi=300)
                
                xlim=[int(i) for i in ax.get_xlim()]
                ylim=[int(i) for i in ax.get_ylim()]
                
                #saves the exact nad precise tiff file
                pilimage = Image.fromarray(img.get_array()[ylim[1]:ylim[0], xlim[0]:xlim[1]])
                pilimage.save(name+'.tiff')
                print('File SAVED @{}'.format(name))
            
            img.set_array(volume[ax.index])
            fig.canvas.draw_idle()
            
            
        

        fig, ax = plt.subplots()
        volume = data
        ax.volume = volume
        ax.index = 1
        ax.set_title('{}/{}  t= {:.2f} s dt= {:.2f} s'.format(ax.index, volume.shape[0], self.time_info[ax.index][0], self.time_info[ax.index][1]))
        
        if source=='diff' or source=='vid':
            img = ax.imshow(volume[ax.index], cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
        else:
            img = ax.imshow(volume[ax.index], cmap='gray')
        fig.canvas.mpl_connect('scroll_event', mouse_scroll)
        fig.canvas.mpl_connect('button_press_event', mouse_click)
        fig.canvas.mpl_connect('key_press_event', button_press)

        
        cb = fig.colorbar(img, ax=ax)
        plt.tight_layout()
        plt.show()
        
        print('''
Buttons "j"/"m" serve to increasing/decreasing contrast 
Button "s" saves the current image as tiff file
Mouse scrolling moves to neighboring frames
Official shortcuts here https://matplotlib.org/users/navigation_toolbar.html
Right mouse button click selects and analysis the NP image
              ''')

    @staticmethod
    def show(img):
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set(xlabel='x [px]', ylabel='y [px]')
        return fig

    def area_show(self, area):
        x, y, width, height = area
        img=self.video[:, :, 1]
        img[y:y + height, x:x + width]+=0.5
        self.show(img)

