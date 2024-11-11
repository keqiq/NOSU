import math
import numpy as np
class Slider:
    
    def __init__(self, data, control, ms_per_beat, velocity):
        self.time = int(data[2])
        self.repeats = int(data[6])
        self.length = float(data[7])
        self.ms_per_beat = ms_per_beat
        self.velocity = velocity
        
        self.control = self.__parse_points(
            [int(data[0]), int(data[1])],
            control
        )
        self.duration_per_slide = self.__calculate_duration()
        self.total_duration = self.duration_per_slide * self.repeats
        self.end_time = self.time + round(self.total_duration)
        self.num_ticks_per_slide = round(self.duration_per_slide / (1000 / 60))
        
        self.ticks = None
        
    @staticmethod
    def __parse_points(start, control):
        results = [start]
        for point in control:
            pos = point.split(':')
            results.append([int(pos[0]), int(pos[1])])
            
        return results
    
    @staticmethod
    def _calculate_points_distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _create_ticks_matrix(self, array):
        self.ticks = np.array(array)
        
    def _calculate_ticks(self):
        results = [[self.control[0][0], self.control[0][1], self.time, 6, self.end_time]]
        for i in range(1, self.num_ticks_per_slide + 1):
            t = i / self.num_ticks_per_slide
            tick_time = int(t * self.duration_per_slide + self.time)
            tick_pos = self._tick_interp(t)
            
            results.append([round(tick_pos[0]), round(tick_pos[1]), tick_time, 7, -1])
        self._create_ticks_matrix(results)
    
    # NEED TO DOUBLE CHECK THIS
    def _calculate_repeats(self):
        repeated_ticks = np.copy(self.ticks)
        for i in range(1, self.repeats):
            copy_tick = np.copy(self.ticks)
            slide_duration = self.ticks[-1][2] - self.time
            
            # WORKING???!?!?! 
            if i % 2 == 1:
                copy_tick[:, :2] = copy_tick[::-1,:2]
                
                # I think this is only needed for bezier sliders
                if type(self).__name__ == 'Bezier':
                    copy_tick[:, 2] = copy_tick[::-1, 2]
                    copy_tick[1:, 2] = np.abs(copy_tick[1:, 2] - copy_tick[:-1, 2])
                    copy_tick[1:, 2] = np.cumsum(copy_tick[1:, 2])
                    copy_tick[:, 2] += self.time
                
                copy_tick[:,2] += slide_duration * i
                
            else:
                copy_tick[:, 2] += slide_duration * i
            
            copy_tick = copy_tick[1:, :]
            
            repeated_ticks = np.vstack((repeated_ticks, copy_tick))
        self.ticks = repeated_ticks
                
            
    def __calculate_duration(self):
        return (self.length / self.velocity) * self.ms_per_beat
    
    def get_ticks(self):
        return self.ticks
    
    def get_end_time(self):
        return self.end_time
    
    
    
        
        