from video_processing import Video
import matplotlib.pyplot as plt
import numpy as np

class RawVideo(Video):
    def __init__(self, folder, file, static_name=None):
        super().__init__(folder, file)
        
        if static_name==None:
            self.static_name=folder + file.replace('raw', 'static_TE')
        self.reference=None
        self._video_ref=None
        self._video_diff=None
        
    def loadData(self):
        super().loadData()
        self.reference=self.loadBinStatic()
        self._video_ref=np.ones(self.video.shape[0:2])
        self._video_diff=np.ones(self.video.shape[0:2])
        
    def loadBinStatic(self):
        code_format = np.float64  # type double
        suffix = '.bin'
        with open(self.static_name+suffix, mode='rb') as fid:
            video = np.fromfile(fid, dtype=code_format)
        video = np.reshape(video, (self.video_stats[1][0],
                                   self.video_stats[1][1]), order='F')

        return video.T
    #refresh reference
    def refref(self):
        sh=self._video.shape
        out=np.zeros(sh)
        
        for i in range(sh[-1]):
            out[:,:,i]=self._video[:,:,i]/self.reference
        self._video_ref=out
        print('refreshed')
        
    def refdiff(self):
        sh=self._video.shape
        out=np.zeros(sh)
        out[:,:,0]=np.zeros(sh[0:2])
        for i in range(1, sh[-1]):
            out[:,:,i]=(self._video[:,:,i]-self._video[:,:,i-1])/self.reference
        self._video_diff=out
        print('refreshed')
        
    def refdifffirst(self):
        sh=self._video.shape
        out=np.zeros(sh)
        out[:,:,0]=np.zeros(sh[0:2])
        for i in range(1, sh[-1]):
#            out[:,:,i]=(self._video[:,:,i]-self._video[:,:,0])
            out[:,:,i]=(self._video[:,:,i]-self._video[:,:,0])/self.reference
        self._video_diff=out
        print('refreshed')
        
    @property
    def video_ref(self):
        return self._video_ref[self.view[1]:self.view[1]+self.view[3], self.view[0]:self.view[0]+self.view[2],: ]
     
    @property
    def video_diff(self):
        return self._video_diff[self.view[1]:self.view[1]+self.view[3], self.view[0]:self.view[0]+self.view[2],: ]
    
    
#    
#        def loadBinVideo(self):
#        code_format = np.float64  # type double
#        suffix = '.bin'
#        with open(self.file_name+suffix, mode='rb') as fid:
#            video = np.fromfile(fid, dtype=code_format)
#        video = np.reshape(video, (self.video_stats[1][0],
#                                   self.video_stats[1][1],
#                                   self.video_stats[1][2]), order='F')
#
#
#        return np.swapaxes(video, 0, 1)