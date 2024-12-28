from .Slider import Slider
import math

"""
Linear slider subclass
"""
class LinearV2(Slider):
    
    def __init__(self, slider_object, ms_per_beat, velocity):
        super().__init__(slider_object, ms_per_beat, velocity)
        _, self.unscaled_length = self._calculate_cumulative_lengths()
        self.scaling_factor = self._calculate_scaling_factor()
        self._scale_control_points()
        self.cumulative_lengths, self.scaled_length = self._calculate_cumulative_lengths()
        self._calculate_ticks()
        if self.repeats > 1: self._calculate_repeats()
    

    """
    Linear tick interpolation function
    Since there can be mulitple segments, call _find_segement_index to find segment which the slider
    ball fall onto at distance determined by t
    """
    def _tick_interp(self, t):
        s = t * self.scaled_length
        i = self._find_segment_index(s)
        l1 = self.cumulative_lengths[i]
        l2 = self.cumulative_lengths[i+1]
        
        alpha = (s - l1) / (l2 - l1)
        x1, y1 = self.control[i]
        x2, y2 = self.control[i+1]
        
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        return x, y