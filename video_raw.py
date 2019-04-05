from video_processing import Video
import matplotlib.pyplot as plt
import numpy as np

class RawVideo(Video):
    def __init__(self, folder, file, static_name=None):
        super().__init__(folder, file)
        
        if static_name==None:
            self.static_name=folder + file.replace('diff', 'static_TE')
        self.reference=None
        
    def loadData(self):
        super().loadData()
        self.reference=self.loadBinStatic()
        
    def loadBinStatic(self):
        code_format = np.float64  # type double
        suffix = '.bin'
        with open(self.static_name+suffix, mode='rb') as fid:
            video = np.fromfile(fid, dtype=code_format)
        video = np.reshape(video, (self.video_stats[1][0],
                                   self.video_stats[1][1],
                                   self.video_stats[1][2]), order='F')


        return np.swapaxes(video, 0, 1)