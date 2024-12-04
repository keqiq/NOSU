from .slider.Linear import Linear
from .slider.Circle import Circle
from .slider.Bezier import Bezier
import numpy as np
import math

class HitObjects():
    
    def __init__(self, map_params, timing_points, hit_objects, hard_rock):
        
        self.map_params = map_params
        self.timing_points = timing_points
        self.hard_rock = hard_rock
        
        self.hit_circles = []
        self.spinners = []
        self.sliders = []
        
        self.all_hit_objects = []
        
        for hit_object in hit_objects:
            self.__parse_hit_object(hit_object)
    
    @staticmethod
    def __check_bit_index(int, index):
        return (int >> index) & 1
    
    def __parse_hit_object(self, hit_object):
        data = hit_object.split(',')
        
        obj = [
            int(data[0]),   # x
            int(data[1]),   # y
            int(data[2]),   # time
            int(data[3]),   # type
            None            # endtime
        ]
        
        # If hit cirlce
        if self.__check_bit_index(obj[3], 0):
            obj[3] = 1
            obj[4] = -1
            self.__parse_hit_circle(obj)
        
        # If slider
        elif self.__check_bit_index(obj[3], 1):
            obj[3] = 6
            self.__parse_slider(obj, data)
        
        # If spinner
        elif self.__check_bit_index(obj[3], 3):
            obj[3] = 12
            obj[4] = int(data[5])
            self.__parse_spinner(obj)
    
    def __parse_hit_circle(self, obj):
        # hard rocks flips object along x-axis so y position is inverted
        if self.hard_rock:
            obj[1] = 384 - (obj[1])
            
        self.hit_circles.append(obj)
        self.all_hit_objects.append(obj)
    
    def __parse_spinner(self, obj):
        # I think spinners alway appear in the center of screen so inverted it would be the same
        self.spinners.append(obj)
        duration = obj[4] - obj[2]
        interval = 1000.0/30.0
        len = math.ceil(duration / interval)
        
        ticks = [[obj[0], obj[1], int(obj[2] + interval * i), 13, -1] for i in range(len)]
        ticks[0][3] = 12
        ticks[0][4] = int(obj[4])
        self.all_hit_objects.extend(ticks)
    
    def __parse_slider(self, obj, data):
        time = obj[2]
        control = data[5].split('|')
        slider_type = control.pop(0)

        base_multiplier = self.map_params.get_slider_multiplier()
        # ms_per_beat = self.timing_points.get_ms_per_beat(time)
        # sv_mulitplier = self.timing_points.get_sv_multiplier(time)
        
        ms_per_beat, sv_multiplier = self.timing_points.get_current_params(time)
        
        velocity = base_multiplier * 100 * sv_multiplier
        
        slider_types = {
            'L' : Linear,
            'P' : Circle,
            'B' : Bezier
        }
        
        slider_class = slider_types[slider_type]
        slider = slider_class(data, control, ms_per_beat, velocity)
        ticks = slider.get_ticks()
        # hard rock, flip all ticks along the x-axis
        if self.hard_rock:
            ticks[:, 1] = 384 - ticks[:, 1]
        self.sliders.append(ticks)
        self.all_hit_objects.extend(ticks)
    
    def get_data(self):
        hit_objects = {
            'hit_circles': np.array(self.hit_circles),
            'spinners': np.array(self.spinners),
            'sliders': np.array(self.sliders, dtype=object)
        }
        data = np.array(self.all_hit_objects).astype(float)
        return hit_objects, data
    
    