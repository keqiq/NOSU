class MapParams():
    
    def __init__(self, params, stack_lenciency):
        
        self.circle_size = self.__parse_param(params[1])
        self.overall_difficulty = self.__parse_param(params[2])
        self.approach_rate = self.__parse_param(params[3])
        self.slider_multiplier = self.__parse_param(params[4])
        self.slider_tick_rate = self.__parse_param(params[5])
        self.stack_leniency = self.__parse_param(stack_lenciency)
        
    @staticmethod
    def __parse_param(param):
        return float(param.split(':')[1])
        
    def get_slider_multiplier(self):
        return self.slider_multiplier
    
    def get_stack_leniency(self):
        return self.stack_leniency
    
    def get_circle_size(self):
        return self.circle_size
    
    def get_preempt(self):
        if self.approach_rate < 5:
            return 1200 + 600 * (5 - self.approach_rate) / 5
        elif self.approach_rate == 5:
            return 1200
        elif self.approach_rate > 5:
            return 1200 - 750 * (self.approach_rate - 5) / 5