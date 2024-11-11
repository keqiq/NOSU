from .Slider import Slider
from .Linear import Linear
import math

class Circle(Slider):
    
    def __init__(self, data, control, ms_per_beat, velocity):
        super().__init__(data, control, ms_per_beat, velocity)
        
        try:
            self.circle = self.__calculate_circle()
            super()._calculate_ticks()
        except ValueError:
            linear = Linear(data, control, ms_per_beat, velocity)
            self.ticks = linear.get_ticks()
        
        if self.repeats > 1: super()._calculate_repeats()
    
    def __calculate_circle(self):
        points = self.control
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]

        # Check if two points have the same y-coordinate (horizontal line case)
        if y1 == y2:
            # If p1 and p2 form a horizontal line, the perpendicular bisector is a vertical line.
            mid12_x = (x1 + x2) / 2  # Midpoint's x-coordinate
            # Use the midpoint of p2 and p3 to compute the slope and determine the center's y-coordinate
            mid23 = ((x2 + x3) / 2, (y2 + y3) / 2)
            slope23_inv = -(x3 - x2) / (y3 - y2)  # Negative reciprocal of slope p2-p3
            center_x = mid12_x
            center_y = slope23_inv * (center_x - mid23[0]) + mid23[1]
        
        elif y2 == y3:
            # If p2 and p3 form a horizontal line, handle similarly
            mid23_x = (x2 + x3) / 2  # Midpoint's x-coordinate
            mid12 = ((x1 + x2) / 2, (y1 + y2) / 2)
            slope12_inv = -(x2 - x1) / (y2 - y1)  # Negative reciprocal of slope p1-p2
            center_x = mid23_x
            center_y = slope12_inv * (center_x - mid12[0]) + mid12[1]

        else:
            # General case where no horizontal lines are present
            mid12 = ((x1 + x2) / 2, (y1 + y2) / 2)
            mid23 = ((x2 + x3) / 2, (y2 + y3) / 2)

            slope12_inv = -(x2 - x1) / (y2 - y1)
            slope23_inv = -(x3 - x2) / (y3 - y2)

            if slope12_inv == slope23_inv:
                raise ValueError("Linear")
            
            center_x = (slope12_inv * mid12[0] - slope23_inv * mid23[0] + mid23[1] - mid12[1]) / (slope12_inv - slope23_inv)
            center_y = slope12_inv * (center_x - mid12[0]) + mid12[1]

        # Calculate radius
        radius = math.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

        return center_x, center_y, radius
    
    def _tick_interp(self, t):
        cx, cy, cr = self.circle
        x1, y1 = self.control[0]
        x2, y2 = self.control[2]
        x3, y3 = self.control[1]

        # Calculate the angles of point1, point2, and point3 relative to the center
        angle1 = math.atan2(y1 - cy, x1 - cx)
        angle2 = math.atan2(y2 - cy, x2 - cx)
        angle3 = math.atan2(y3 - cy, x3 - cx)

        # Ensure that the angles are within the correct range
        if angle2 < angle1:
            angle2 += 2 * math.pi
        if angle3 < angle1:
            angle3 += 2 * math.pi

        # Check if angle3 lies between angle1 and angle2 (shorter arc direction)
        if angle1 <= angle3 <= angle2:
            # Interpolate between angle1 and angle2
            interpolated_angle = (1 - t) * angle1 + t * angle2
        else:
            # If angle3 is not between angle1 and angle2, take the longer arc
            if angle3 > angle2:
                angle3 -= 2 * math.pi  # Normalize the angle to the same range
                # Interpolate between angle1 and angle3
            interpolated_angle = (1 - 2 * t) * angle1 + 2 * t * angle3

        # Calculate the position of the interpolated point on the arc
        x = cx + cr * math.cos(interpolated_angle)
        y = cy + cr * math.sin(interpolated_angle)

        return x, y