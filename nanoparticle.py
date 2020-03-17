import tools as tl
import numpy as np

class NanoParticle():
    def __init__(self, np_id, positions, peak = None, masks = None, k_diff = 10, method = 'beta'):
        self.id = np_id
        self.color = tl.random_color()
        

        if method == 'beta':
            self.positions = positions
            self.masks = masks
            self.peak = peak
            
        elif method == 'alpha':
            self.positions = [(positions[0] + i, positions[1], positions[2]) for i in range(-k_diff//2, k_diff//2)]
            self.masks = masks
            self.peak = positions[0] 
        
        self.first_frame = self.positions[0][0]
        self.last_frame = self.positions[-1][0]         
        
    def position_yx(self, frame):
        if frame > self.last_frame:
            raise Exception('The nanoparticle does not occur in this frame.')
        return self.positions[frame-self.first_frame][:-3:-1]
    
    def last_position_yx(self):
        return self.position_yx(self.last_frame)
    
    def first_position_yx(self):
        return self.position_yx(self.first_frame)
    
    @property
    def mask_for_characterization(self):
        mask = np.squeeze(self.masks[self.peak-self.first_frame])
        time = np.full((len(mask), 1), self.peak, dtype=int)
        print(len(mask))
        print(len(time))
        return np.concatenate((time, mask))
        