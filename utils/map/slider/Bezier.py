from .Slider import Slider
import numpy as np
class Bezier(Slider):
    
    def __init__(self, data, control, ms_per_beat, velocity):
        super().__init__(data, control, ms_per_beat, velocity)
        
        self.sections = self.__separate_sections()
        self.sections_lengths = self.__caclulate_section_lengths()
        
        self.__calculate_ticks()
        if self.repeats > 1: super()._calculate_repeats()

    @staticmethod
    def binomialCoefficient(n, k):
        if k < 0 or k > n:   return 0
        if k == 0 or k == n: return 1

        k = min(k, n - k)  # Take advantage of geometry
        c = 1

        for i in range(k):
            c *= (n - i) / (i + 1)

        return c
    
    def __bezier_interp(self, control, t):
        bx, by, n = 0, 0, len(control) - 1

        # Linear
        if (n == 1):
            bx = (1 - t) * control[0][0] + t * control[1][0]
            by = (1 - t) * control[0][1] + t * control[1][1]

        # Quadratic
        elif (n == 2):
            bx = (1 - t) * (1 - t) * control[0][0] + 2 * (1 - t) * t * control[1][0] + t * t * control[2][0]
            by = (1 - t) * (1 - t) * control[0][1] + 2 * (1 - t) * t * control[1][1] + t * t * control[2][1]

        # Cubic
        elif (n == 3):
            bx = (1 - t) * (1 - t) * (1 - t) * control[0][0] + 3 * (1 - t) * (1 - t) * t * control[1][0] + 3 * (1 - t) * t * t * control[2][0] + t * t * t * control[3][0]
            by = (1 - t) * (1 - t) * (1 - t) * control[0][1] + 3 * (1 - t) * (1 - t) * t * control[1][1] + 3 * (1 - t) * t * t * control[2][1] + t * t * t * control[3][1]

        # Generalized equation
        else:
            for i in range(n + 1):
                    bx += self.binomialCoefficient(n, i) * pow(1 - t, n - i) * pow(t, i) * control[i][0]
                    by += self.binomialCoefficient(n, i) * pow(1 - t, n - i) * pow(t, i) * control[i][1]
                    
        return bx, by
                
    def __separate_sections(self):
        points = self.control
        results = []
        current_section = [points[0]]
        
        for i in range(1, len(points)):
            
            if points[i] == points[i-1]:
                results.append(current_section)
                current_section = [points[i]]
                
            else :
                current_section.append(points[i])
                
        results.append(current_section)
        return results
    
    def __caclulate_section_lengths(self, num_segments = 50):
        results = []
        
        for section in self.sections:
            prev_point = section[0]
            section_distance = 0
            
            for i in range(num_segments + 1):
                point = self.__bezier_interp(section, i / num_segments)
                section_distance += super()._calculate_points_distance(prev_point, point)
                
                prev_point = point
            
            results.append(section_distance)
        return results
            
    def __calculate_ticks(self):
        results = [[self.control[0][0], self.control[0][1], self.time, 6, self.end_time]]
        total_length = sum(self.sections_lengths)
        total_duration = 0
        
        for i in range(len(self.sections)):
            section = self.sections[i]
            section_length = self.sections_lengths[i]
            
            section_duration = (section_length / total_length) * self.duration_per_slide
            section_num_ticks = round(section_duration / (1000 / 60))
            
            # Weird sliders with too many control points and or tiny section lengths may need this check
            # In this case, just append a tick at the middle of the control points
            if section_num_ticks < 1:
                t = 0.5
                tick_time = int(t * section_duration + total_duration + self.time)
                tick_pos = self.__bezier_interp(section, t)
                results.append([round(tick_pos[0]), round(tick_pos[1]), tick_time, 7, -1])
            else:
                for j in range(1, section_num_ticks + 1):
                    t = j / section_num_ticks
                    tick_time = int(t * section_duration + total_duration + self.time)
                    tick_pos = self.__bezier_interp(section, t)
                    
                    results.append([round(tick_pos[0]), round(tick_pos[1]), tick_time, 7, -1])
            
            total_duration += section_duration
        super()._create_ticks_matrix(results)