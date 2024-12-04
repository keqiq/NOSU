from .Slider import Slider

class Linear(Slider):
    
    def __init__(self, data, control, ms_per_beat, velocity):
        super().__init__(data, control, ms_per_beat, velocity)
        
        super()._calculate_ticks()
        if self.repeats > 1: super()._calculate_repeats()
    
    def _tick_interp(self, t):
        x1, y1 = self.control[0]
        x2, y2 = self.control[-1]
        
        # Linear interpolation formula
        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2
        
        return x, y
        
            
            
            