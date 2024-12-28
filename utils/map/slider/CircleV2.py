from .Slider import Slider
from .Linear import Linear
import math

"""
Circle slider subclass
"""
class CircleV2(Slider):
    
    def __init__(self, slider_object, ms_per_beat, velocity):
        super().__init__(slider_object, ms_per_beat, velocity)
        
        try:
            self.cx, self.cy, self.r = self.__calculate_circle()
            self.arc_angle, self.is_ccw = self.__determine_arc_direction()
            self.unscaled_length = self.r * self.arc_angle
            self.scaling_factor = self._calculate_scaling_factor()
            self._scale_circle()
            
            self._calculate_ticks()
            if self.repeats > 1: self._calculate_repeats()
        except ValueError:
            # This Value Error is thrown when the circle slider is flat and may be considered linear
            linear = Linear(slider_object, ms_per_beat, velocity)
            self.ticks = linear.get_ticks()
    
    """    
    This function takes in 3 control points and calculates the center and radius of circle
    If points are collinear (flat) then falls back to linear slider
    """ 
    def __calculate_circle(self):
        p1, p2, p3 = self.control
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Handle degenerate vertical/horizontal cases safely by checking denominators
        denom = 2 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        if abs(denom) < 1e-12:
            # Points are likely collinear or too close
            raise ValueError("Linear")
        
        # Formula for circle center (cx, cy)
        ux = ((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2)) / denom
        uy = ((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1)) / denom

        cx = ux
        cy = uy

        # Radius
        radius = math.sqrt((cx - x1)**2 + (cy - y1)**2)
        return cx, cy, radius
    
    """
    Function to determine arc direction cw or ccw and compute arc angle
    """
    def __determine_arc_direction(self):
        p1, p2, p3 = self.control
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Get angles from each control point to circle center
        angle1 = math.atan2(y1 - self.cy, x1 - self.cx) % (2*math.pi)
        angle2 = math.atan2(y2 - self.cy, x2 - self.cx) % (2*math.pi)
        angle3 = math.atan2(y3 - self.cy, x3 - self.cx) % (2*math.pi)
        
        self.angle1 = angle1
        
        # Counter-clockwise angle from point1 to point3
        angle13_ccw = (angle3 - angle1) % (2 * math.pi)
        # Clockwise angle from point1 to point3
        angle13_cw = (angle1 - angle3) % (2 * math.pi)
        
        # Determine which arc contains the second point
        angle12_ccw = (angle2 - angle1) % (2 * math.pi)
        if angle12_ccw <= angle13_ccw:
            # Arc is counter-clockwise
            return angle13_ccw, True
        else:
            # Arc is clockwise
            return angle13_cw, False
        
    """
    Function to scale the points according to slider length in .osu map file
    """
    def _scale_circle(self):
        x1, y1 = self.control[0]
        
        # Translate so start point is origin
        self.cx -= x1
        self.cy -= y1

        # Scale
        # Circle center is also scaled this way the starting point does not change
        self.cx *= self.scaling_factor
        self.cy *= self.scaling_factor
        self.r *= self.scaling_factor

        # Translate back
        self.cx += x1
        self.cy += y1
    
    """
    Circle slider tick interpolation function
    """
    def _tick_interp(self, t):
        # Use the arc direction
        if self.is_ccw:
            # Counter-clockwise
            interpolated_angle = (self.angle1 + t * self.arc_angle) % (2 * math.pi)
        else:
            # Clockwise
            interpolated_angle = (self.angle1 - t * self.arc_angle) % (2 * math.pi)

        x = self.cx + self.r * math.cos(interpolated_angle)
        y = self.cy + self.r * math.sin(interpolated_angle)
        return x, y