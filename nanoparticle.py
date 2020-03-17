import tools as tl


class NanoParticle():
    def __init__(self, np_id, positions, masks = None, k_diff = 10, method = 'beta'):
        self.id = np_id
        self.color = tl.random_color()

        if method == 'beta':
            self.positions = positions
            self.masks = masks
            
        elif method == 'alpha':
            self.positions = [(positions[0] + i, positions[1], positions[2]) for i in range(-k_diff//2, k_diff//2)]
            self.masks = masks
        
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
        