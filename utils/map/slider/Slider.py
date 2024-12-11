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
        self.max_distance = 0
        
        self.ticks = None
    
    # Splits the control points from the .osu map format into array
    @staticmethod
    def __parse_points(start, control):
        results = [start]
        for point in control:
            pos = point.split(':')
            results.append([int(pos[0]), int(pos[1])])
            
        return results
    
    def _calculate_scaling_factor(self):
        return self.length / self.unscaled_length
    
    @staticmethod
    def _calculate_points_distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # To determine if a slider is a buzz slider/short slider or a normal slider
    # Calculate the max distance between slider head (start) and each tick
    def _update_max_distance(self, tick_pos):
        distance = self._calculate_points_distance(self.control[0], tick_pos)
        if distance > self.max_distance:
            self.max_distance = distance
    
    # Called after all tick interpolation and appends a buzz slider indicator
    def _create_ticks_matrix(self, ticks):
        num_rows = len(ticks)
        is_buzz = 1 if self.max_distance < 90 else 0
        col_indicator = np.full((num_rows, 1), is_buzz)
        self.ticks = np.hstack((np.array(ticks), col_indicator))
        
    # Calculates the position and time of slider ticks
    # Calls subclass _tick_interp for different curve types
    # For bezier sliders this superclass method is not used due to completely different parsing
    def _calculate_ticks(self):
        results = [[self.control[0][0], self.control[0][1], self.time, 6, self.end_time]]
        for i in range(1, self.num_ticks_per_slide + 1):
            t = i / self.num_ticks_per_slide
            tick_time = int(t * self.duration_per_slide + self.time)
            tick_pos = self._tick_interp(t)
            
            results.append([round(tick_pos[0]), round(tick_pos[1]), tick_time, 7, -1])
            self._update_max_distance(tick_pos)
        self._create_ticks_matrix(results)
    
    # Calculates repeated slides
    # Since speed is constant during a slider, a repeat is just the first slide reversed
    # More generally, first repeat (odd) is reversed, second repeat (even) would be the same as the first slide
    def _calculate_repeats(self):
        repeated_ticks = np.copy(self.ticks)
        slide_duration = self.ticks[-1][2] - self.time

        # Separate positions and times for clarity
        original_positions = self.ticks[:, :2]
        original_times = self.ticks[:, 2]

        for i in range(1, self.repeats):
            # For odd repeats, reverse positions
            if i % 2 == 1:
                new_positions = original_positions[::-1, :]
            else:
                new_positions = np.copy(original_positions)

            # Times just increment by slide_duration * i
            new_times = original_times + slide_duration * i

            # Combine them back
            copy_tick = np.column_stack((new_positions, new_times, self.ticks[:, 3], self.ticks[:, 4], self.ticks[:, 5]))

            # Remove the first tick to avoid duplication
            copy_tick = copy_tick[1:, :]

            repeated_ticks = np.vstack((repeated_ticks, copy_tick))

        self.ticks = repeated_ticks
            
    def __calculate_duration(self):
        return (self.length / self.velocity) * self.ms_per_beat
    
    def get_ticks(self):
        return self.ticks
    
    def get_end_time(self):
        return self.end_time
    
    
    
        
        