from .Slider import Slider
# Bezier slider subclass similar to how bezier sliders are represented in game
# The bezier segment(s) are approximated with polyline(s)
class BezierV2(Slider):
    def __init__(self, slider_object, ms_per_beat, velocity):
        super().__init__(slider_object, ms_per_beat, velocity)
        self.segments = self._get_bezier_segments()
        self.flattened_path = self._flatten_all_segments()
        self.cumulative_lengths, self.total_length = self._calculate_cumulative_lengths()
        
        self.__calculate_ticks()
        if self.repeats > 1: super()._calculate_repeats()
        
    # Separates sections based on consecutive control points with the same coordinates
    def _get_bezier_segments(self):
        segments = []
        current_segment = [self.control[0]]

        for i in range(1, len(self.control)):
            if self.control[i] == self.control[i - 1]:
                segments.append(current_segment)
                current_segment = [self.control[i]]
            else:
                current_segment.append(self.control[i])

        segments.append(current_segment)
        return segments
    
    # Using De Casteljau's algorithm to split the Bezier curve into two halves.
    @staticmethod
    def _subdivide_bezier(cp):
        n = len(cp) - 1
        midpoints = [list(cp)]
        for r in range(1, n + 1):
            midpoints.append([])
            for i in range(n - r + 1):
                x = 0.5 * (midpoints[r - 1][i][0] + midpoints[r - 1][i + 1][0])
                y = 0.5 * (midpoints[r - 1][i][1] + midpoints[r - 1][i + 1][1])
                midpoints[r].append([x, y])

        left = [midpoints[i][0] for i in range(n + 1)]
        right = [midpoints[n - i][i] for i in range(n + 1)]
        return left, right
    
    # Calculates the perpendicular distance from a point to a line.
    @staticmethod
    def _point_line_distance(x0, y0, x1, y1, x2, y2):
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        return numerator / denominator
    
    # Computes the perpendicular distance from control points to the line
    # between the first and last control points
    # If within tolerance then the section is flat enough
    def _is_flat_enough(self, cp, tol):
        x0, y0 = cp[0]
        x3, y3 = cp[-1]

        # Check for degenerate chord
        if x0 == x3 and y0 == y3:
            # If there are no intermediate points, the curve is flat enough
            if len(cp) <= 2:
                return True
            else:
                # Calculate maximum distance from control points to (x0, y0)
                max_distance = max(
                    self._calculate_points_distance((xi, yi), (x0, y0))
                    for xi, yi in cp[1:-1]
                )
                return max_distance < tol

        max_distance = 0
        for xi, yi in cp[1:-1]:
            distance = self._point_line_distance(xi, yi, x0, y0, x3, y3)
            if distance > max_distance:
                max_distance = distance

        return max_distance < tol
    # Recursively subdivides segments into a polyline approximating the bezier curve
    def _flatten_bezier(self, cp, epsilon):
        def recursive_flatten(cp, tol, output):
            if self._is_flat_enough(cp, tol):
                output.append(cp[-1])
            else:
                left, right = self._subdivide_bezier(cp)
                recursive_flatten(left, tol, output)
                recursive_flatten(right, tol, output)
            
        output = [cp[0]]
        recursive_flatten(cp, epsilon, output)
        return output

    # Calls _flatten_bezier on each segment
    def _flatten_all_segments(self, epsilon=0.5):
        flattened_path = []

        for segment in self.segments:
            flattened_segment = self._flatten_bezier(segment, epsilon)
            # Exclude the first point if it's not the starting point to avoid duplicates
            if flattened_path and flattened_segment[0] == flattened_path[-1]:
                flattened_path.extend(flattened_segment[1:])
            else:
                flattened_path.extend(flattened_segment)

        return flattened_path
    
    # Calculates cumulative length after each segment and total length of all segments
    def _calculate_cumulative_lengths(self):
        lengths = [0]
        total_length = 0

        for i in range(1, len(self.flattened_path)):
            p1 = self.flattened_path[i - 1]
            p2 = self.flattened_path[i]
            segment_length = self._calculate_points_distance(p1, p2)
            total_length += segment_length
            lengths.append(total_length)

        return lengths, total_length
    
    def _calculate_tick_distance(self):
        tick_interval_ms = 1000 / 60  # 16.6667 ms per tick for 60fps
        velocity_per_ms = self.velocity / self.ms_per_beat  # osu pixels per millisecond

        tick_distance = velocity_per_ms * tick_interval_ms
        return tick_distance
    
    # Binary search to find the segment index where the cumulative length exceeds the distance
    def _find_segment_index(self, distance):
        left = 0
        right = len(self.cumulative_lengths) - 1

        while left < right:
            mid = (left + right) // 2
            if self.cumulative_lengths[mid] < distance:
                left = mid + 1
            else:
                right = mid

        return max(0, left - 1)
    
    def __calculate_ticks(self):
        tick_distance = self._calculate_tick_distance()
        results = []
        next_tick_distance = tick_distance
        total_duration = self.duration_per_slide
        slider_start_time = self.time

        # Add the slider head (the starting point)
        results.append([self.flattened_path[0][0], self.flattened_path[0][1], self.time, 6, self.end_time])

        # Place ticks along the path
        while next_tick_distance < self.total_length:
            # Find the segment where the next tick lies
            index = self._find_segment_index(next_tick_distance)

            # Calculate the exact position of the tick
            p1 = self.flattened_path[index]
            p2 = self.flattened_path[index + 1]
            l1 = self.cumulative_lengths[index]
            l2 = self.cumulative_lengths[index + 1]

            # Interpolate to find the position
            alpha = (next_tick_distance - l1) / (l2 - l1)
            tick_x = p1[0] + alpha * (p2[0] - p1[0])
            tick_y = p1[1] + alpha * (p2[1] - p1[1])

            # Calculate tick time based on tick interval
            tick_time = int(slider_start_time + (next_tick_distance / self.total_length) * total_duration)

            results.append([round(tick_x), round(tick_y), tick_time, 7, -1])
            next_tick_distance += tick_distance
            self._update_max_distance([tick_x, tick_y])
        
        # Adding slider end as ticks are calculated by distance and not time
        # The end of the slider at start_time + total duration may be missed in some cases
        # Missing the slider end is compounded when there are multiple repeats
        # This usually is only a problem when the slider is very short, fast and repeats a lot
        if len(results) == 1:
            end_x, end_y = self.flattened_path[-1]
            end_tick_time = slider_start_time + total_duration
            results.append([round(end_x), round(end_y), end_tick_time, 7, -1])

        super()._create_ticks_matrix(results)