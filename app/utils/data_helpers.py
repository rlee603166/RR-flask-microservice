import numpy as np
    
def center_pts(kps_arr):
    num_frames = len(kps_arr)
    
    centroid_x = 0
    centroid_y = 0
    num_pts = 0
    
    last_valid_xy = None
    
    for i in range(num_frames):
        keypoints = kps_arr[i][0]
        num_pts += len(keypoints)
        centroid_x += sum(keypoints[:, 0])
        centroid_y += sum(keypoints[:, 1])


    centroid_x /= num_pts
    centroid_y /= num_pts
    
    centered_keypoints = []
    

    for i in range(num_frames):
        front_xy = kps_arr[i][0]

        if len(front_xy) < 17:
            if last_valid_xy is not None:
                front_xy = last_valid_xy
            else:
                front_xy = np.insert(front_xy, 10, [[front_xy[9, 0] - 1, front_xy[9, 1] - 1]], axis=0)
                
        else:
            last_valid_xy = front_xy 
        
        centered_kps = front_xy - np.array([0, 0])
        x = centered_kps[:, 0]
        y = centered_kps[:, 1]
        x_normalized = (x - np.min(y)) / (np.max(y) - np.min(y))
        y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
        y_inverted = np.max(y_normalized) - y_normalized


        x_centered = x_normalized - np.max(x_normalized) + ((np.max(x_normalized) - np.min(x_normalized)) / 2)

        normalized_keypoints = np.column_stack((x_centered, y_inverted))
        centered_keypoints.append(normalized_keypoints)
    
    return centered_keypoints

def align_vids(front, f_impact, back, b_impact):
    f_impact_frame = int(len(front) * f_impact)
    b_impact_frame = int(len(back) * b_impact)

    difference = f_impact_frame - b_impact_frame

    if difference > 0:
        front = front[difference::].copy()
    elif difference < 0:
        back = back[abs(difference)::].copy()
        
    impact_frame = f_impact_frame
    
    if len(front) < len(back):
        back = back[:len(front)]
        impact_frame = f_impact_frame
    elif len(back) < len(front):
        front = front[:len(back)]
        impact_frame = b_impact_frame

    return front, back, impact_frame

def recursive_convert_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [recursive_convert_to_list(item) for item in data]
    elif isinstance(data, dict):
        return {key: recursive_convert_to_list(value) for key, value in data.items()}
    return data