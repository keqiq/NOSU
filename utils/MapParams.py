class MapParams():
    
    def __init__(self, params):
        
        self.circle_size = self.__parse_param(params[1])
        self.overall_difficulty = self.__parse_param(params[2])
        self.approach_rate = self.__parse_param(params[3])
        self.slider_multiplier = self.__parse_param(params[4])
        self.slider_tick_rate = self.__parse_param(params[5])
        
    @staticmethod
    def __parse_param(param):
        return float(param.split(':')[1])
        
    def get_slider_multiplier(self):
        return self.slider_multiplier