from .Slider import Slider

class Linear(Slider):
    
    def __init__(self, slider_object, ms_per_beat, velocity):
        super().__init__(slider_object, ms_per_beat, velocity)
        self.unscaled_length = super()._calculate_points_distance(self.control[0], self.control[-1])
        self.scaling_factor = super()._calculate_scaling_factor()
        self._scale_control_points()
        super()._calculate_ticks()
        if self.repeats > 1: super()._calculate_repeats()
    
    # Function to scale the points according to slider length in .osu map file
    def _scale_control_points(self):
        # Linear slider has two points
        x1, y1 = self.control[0]
        x2, y2 = self.control[-1]

        # Compute the direction vector from start to end
        dx = x2 - x1
        dy = y2 - y1

        # Scale the direction vector
        dx *= self.scaling_factor
        dy *= self.scaling_factor

        # Update the end control point
        self.control[-1] = [x1 + dx, y1 + dy]
    
    def _tick_interp(self, t):
        x1, y1 = self.control[0]
        x2, y2 = self.control[-1]
        
        # Linear interpolation formula
        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2
        
        return x, y