import numpy as np

class TimingPoints():
    
    def __init__(self, timing_points):
        
        self.uninherited_timing_points = None
        self.inherited_timing_points = None
        
        self.__parse_timing_point(timing_points)
        
    
    def __parse_timing_point(self, timing_points):
        utps = []
        itps = []
        for timing_point in timing_points:
            tp_data = timing_point.split(',')
            if (tp_data[6] == '1'):
                utps.append(
                    [int(tp_data[0]),float(tp_data[1])]
                    )
            else:
                itps.append(
                    [int(tp_data[0]), -100 / float(tp_data[1])]
                    )
        
        self.uninherited_timing_points = np.array(utps)
        self.inherited_timing_points = np.array(itps)
        
    def __get_timing_point(self, time):
        # if type == "uninherited":
        #     tp = self.uninherited_timing_points
        # elif type == 'inherited':
        #     # Check for case where there is no inherited timing point at current time
        #     tp = self.inherited_timing_points
        
        # splice = tp[:, 0]
        # last_tp_index = np.searchsorted(splice, time, side='right') - 1
        # return tp[last_tp_index]
        
        utps_splice = self.uninherited_timing_points[:, 0]
        itps_splice = self.inherited_timing_points[:, 0]
        
        last_utps_idx = max(np.searchsorted(utps_splice, time, side='right') - 1, 0)
        last_itps_idx = max(np.searchsorted(itps_splice, time, side='right') - 1, 0)
        
        return self.uninherited_timing_points[last_utps_idx], self.inherited_timing_points[last_itps_idx]
    
    def get_ms_per_beat(self, time):
        return self.__get_timing_point(time, 'uninherited')[1]
    
    def get_sv_multiplier(self, utps, time):
        return self.__get_timing_point(time, 'inherited')[1]
    
    def get_current_params(self, time):
        utp, itp = self.__get_timing_point(time)
        
        ms_per_beat = utp[1]
        sv_multiplier = itp[1]
        
        # Case when last timing point used is uninherited
        if itp[0] < utp[0] or itp[0] > time:
            sv_multiplier = 1
            
        return ms_per_beat, sv_multiplier
        
    
