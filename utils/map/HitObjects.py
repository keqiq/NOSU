from .slider.Linear import Linear
from .slider.CircleV2 import CircleV2
from .slider.BezierV2 import BezierV2
import numpy as np
import math

class HitObjects():
    
    def __init__(self, map_params, timing_points, hit_objects, hard_rock):
        
        self.base_multiplier = map_params.get_slider_multiplier()
        # self.map_params = map_params
        self.timing_points = timing_points
        self.hard_rock = hard_rock
        
        # Used by visualizer
        self.hit_circles = []
        self.spinners = []
        self.sliders = []
        
        # Used by model
        self.result_hit_objects = []
        
        self.radius = (54.4 - 4.48 * map_params.get_circle_size())
        self.preempt = map_params.get_preempt()
        self.stack_dist = 3
        self.stack_leniency = map_params.get_stack_leniency()
        self.stack_counter, self.parsed_hit_objects = self.__parse_and_stack_count(hit_objects)
        # I'm not sure what the exact offset is but this work fine for now
        self.stack_offset = 3 * (self.radius / 36.48)
        
        # for hit_object in hit_objects:
        #     self.__parse_hit_object(hit_object)
        
        self.__process_hit_objects()
            
    # Function to check the type of hit objects based on bit index       
    @staticmethod
    def __check_bit_index(int, index):
        return (int >> index) & 1
    
    # Function to convert string type hitobjects in .osu file into objects
    def __parse_hit_object(self, hit_object):
        data = hit_object.split(',')
        
        obj = {
            'x': int(data[0]),
            'y': int(data[1]),
            'time': int(data[2]),
            'type': int(data[3]),
            'end_time': -1,
            'buzz': 0,
            'slider_data': None,
        }
        
        if self.__check_bit_index(obj['type'], 0):
            obj['type'] = 1
        elif self.__check_bit_index(obj['type'], 1):
            obj['type'] = 6
            slider_parts = data[5].split('|')
            slider_data = {
                'shape':    slider_parts[0],
                'controls': slider_parts[1:],   
                'repeats':  int(data[6]),
                'length':   float(data[7])
            }
            obj['slider_data'] = slider_data
        elif self.__check_bit_index(obj['type'], 3):
            obj['type'] = 12
            obj['end_time'] = int(data[5])
        return obj
    
    # Create array of stack counters (not perfect but i can't find more documentation)
    # Stacking occurs on hit circles and slider heads
    def __parse_and_stack_count(self, hit_objects):
        stack_counter = [0] * len(hit_objects)
        parsed_hit_objects = [None] * len(hit_objects)
        # The loop is performed in reverse order
        # If the ith object is within stacking distance and time threshold of i+1th object
        # Increment stack counter by 1 of i+1th object's stack counter
        for i in range(len(hit_objects)-1, -1, -1):
            curr_obj = self.__parse_hit_object(hit_objects[i])
            parsed_hit_objects[i] = curr_obj
            
            # Spinner check
            if curr_obj['type'] == 12:
                continue
            
            # Calculating time threshold based on current beat duration at time ti
            # ms_per_beat, _ = self.timing_points.get_current_params(curr_obj['time'])
            # print(ms_per_beat)
            time_threshold = self.preempt * self.stack_leniency
            
            # Look ahead at future objects to determine if stacking should occur
            for j in range(i + 1, len(hit_objects)):
                next_obj = parsed_hit_objects[j]
                
                # Check if next_obj is within time threshold
                if (next_obj['time'] - curr_obj['time']) > time_threshold:
                    break
                
                # Also perfrom spinner check
                if curr_obj['type'] == 12:
                    continue
                
                # Compute distance between curr and next obj
                dx = next_obj['x'] - curr_obj['x']
                dy = next_obj['y'] - curr_obj['y']
                dist = (dx*dx + dy*dy) ** 0.5
                
                # Stacking occurs if the distance falls below the stacking distance threshold
                if dist < self.stack_dist:
                    curr_stack_count = stack_counter[j] + 1
                    
                    # This ensures that the stack counter is never decremented
                    # Strange check but there could be potential edge case scenarios
                    # I cannot explain these scenarios well enough with words
                    if curr_stack_count > stack_counter[i]:
                        stack_counter[i] = curr_stack_count
                        
        return stack_counter, parsed_hit_objects
    
    def __to_array(self, hit_object):
        obj_array = [
            hit_object['x'],
            hit_object['y'],
            hit_object['time'],
            hit_object['type'],
            hit_object['end_time'],
            hit_object['buzz']
        ]
        return obj_array
    
    def __process_hit_circle(self, hit_circle):
        if self.hard_rock:
            hit_circle['y'] = 384 - hit_circle['y']
         
        self.hit_circles.append(self.__to_array(hit_circle))
        self.result_hit_objects.append(self.__to_array(hit_circle))
        
    def __process_spinner(self, spinner):
        self.spinners.append(self.__to_array(spinner))
        
        duration = spinner['end_time'] - spinner['time']
        interval = 1000.0 / 30.0
        num_ticks = math.ceil(duration / interval)
        
        spinner_ticks = [[spinner['x'], spinner['y'], int(spinner['time'] + interval * i), 13, -1, 0] for i in range(num_ticks)]
        spinner_ticks[0][3] = 12
        spinner_ticks[0][4] = int(spinner['end_time'])
        self.result_hit_objects.extend(spinner_ticks)
        
    def __process_slider(self, slider):
        slider_shapes = {
            'L' : Linear,
            'P' : CircleV2,
            'B' : BezierV2
        }
        if slider['time'] == 369920:
            pass
        ms_per_beat, sv_multiplier = self.timing_points.get_current_params(slider['time'])
        velocity = self.base_multiplier * 100 * sv_multiplier
        
        slider_class = slider_shapes[slider['slider_data']['shape']]
        slider_object = slider_class(slider, ms_per_beat, velocity)
        ticks = slider_object.get_ticks()
        
        if self.hard_rock:
            ticks[:, 1] = 384 - ticks[:, 1]
        
        self.sliders.append(ticks)
        self.result_hit_objects.extend(ticks)
        
    # Function to turn parsed hit objects into format used by visualizer and model
    # A kind of wrapper function which calls type specific processing function
    # Also applies stacking offsets
    def __process_hit_objects(self):
        processors = {
            1: self.__process_hit_circle,
            6: self.__process_slider,
            12: self.__process_spinner
        }
        for i in range(len(self.parsed_hit_objects)):
            hit_object = self.parsed_hit_objects[i]
            
            object_offset = self.stack_counter[i] * self.stack_offset
            hit_object['x'] -= object_offset
            hit_object['y'] -= object_offset
            
            processor_func = processors.get(hit_object['type'])
            processor_func(hit_object)
    
    def get_data(self):
        # hit_objects is used by the visualizer
        for_visualizer = {
            'hit_circles': np.array(self.hit_circles),
            'spinners': np.array(self.spinners),
            'sliders': np.array(self.sliders, dtype=object)
        }
        # data is used during sequence generation
        for_model = np.array(self.result_hit_objects).astype(float)
        return for_visualizer, for_model