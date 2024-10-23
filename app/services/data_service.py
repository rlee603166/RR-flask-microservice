import numpy as np
from app.utils.data_helpers import *

class DataProcessor:
    
    def post_process(self, front, front_impact_time, back, back_impact_time):
        front = center_pts(front)
        back = center_pts(back)
        
        front = recursive_convert_to_list(front)
        back = recursive_convert_to_list(back)
        
        front_kps, back_kps, impact_frame = align_vids(front, float(front_impact_time), back, float(back_impact_time))
        
        return front_kps, back_kps, impact_frame
    
        