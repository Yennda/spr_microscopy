import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import skvideo.io
import skvideo.datasets
import scipy.misc

class VideoLoad(object):

    def __init__(self, file_name):
       
        self.file_name = file_name
        self.video_stats = None
        self._video = None
        self.view = None
        self.rng= [-1, 1]

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
    
    def export(self, name, auto=True):
        data=np.swapaxes(np.swapaxes(self.video,0,2),1,2)  
        
        if auto:
            data-=data.min()
            data*=256/data.max()
        else:
            data-=self.rng[0]
            data*=256/(self.rng[1]-self.rng[0])

        
        
        if self.video_stats[1][2] == 1:
            scipy.misc.toimage(data[0, :, :]).save(name+'.png')
        else:
            writer = skvideo.io.FFmpegWriter(name+'.mp4')
            for i in range(self.video_stats[1][2]):
                writer.writeFrame(data[i, :, :])
            writer.close()

    def frame(self, arg=1):

        if type(arg) == int:
            data = self.video[:, :, arg]
        else:
            data = np.array(arg)
        fig, ax = plt.subplots()
        ax.imshow(data, cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
        ax.set(xlabel='x [px]', ylabel='y [px]')
        return fig
    
    
    def explore(self):
        fig, ax = plt.subplots()
        data=self.video[:,:,0]
        ax.set(xlabel='x [px]', ylabel='y [px]')
        ax.imshow(data, cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
        plt.show()
        
        i=0;
        com='n'
        while com!='q':
            com=input('in:')
            if com=='a':
                i+=1
            elif com=='d':
                i-=1
            elif com=='w':
                i+=10
            elif com=='s':
                i-=10
                
            data=self.video[:,:,i]
            
            ax.imshow(data, cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
            ax.set(xlabel='x [px]', ylabel='y [px]')
            plt.show()
            
        
        
        
    
    def play(self, fr=[0, -1], delta=0.2):
    
        fig, ax = plt.subplots()
    
        
        video_load=self.video[: ,: , fr[0]:fr[1]]
    
        for i in range(0, fr[1]-fr[0]):
            ax.set_title(str(i)+'/'+str(fr[1]-fr[0]))
            ax.imshow(video_load[:, :, i], cmap='gray', vmin=self.rng[0], vmax=self.rng[1])
            plt.pause(delta)
        return fig
    

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

