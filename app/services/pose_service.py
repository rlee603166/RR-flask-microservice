import numpy as np
import tensorflow as tf
from app.utils.pose_helpers import _keypoints_and_edges_for_display, determine_crop_region, init_crop_region, run_inference

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

MIN_CROP_KEYPOINT_SCORE = 0.2

class PoseDetection:

    def __init__(self):
        self.model_path = 'app/services/movenet-tensorflow2-singlepose-thunder-v4'
        self.model_name = 'movenet_thunder'
        self.model = tf.saved_model.load(self.model_path)
        self.input_size = 256
        
    def movenet(self, input_image):
        movenet = self.model.signatures['serving_default']
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = movenet(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores
    
    def predict(self, gif):
        num_frames, org_height, org_width, _ = gif.shape
        crop_region = init_crop_region(org_height, org_width)
        kps_edges = []
        
        for i in range(num_frames):
            kps_scores = run_inference(
                self.movenet,
                gif[i, :, :, :],
                crop_region,
                crop_size=[self.input_size, self.input_size]
            )
            kps_edges.append(_keypoints_and_edges_for_display(
                kps_scores,
                org_height,
                org_width
            ))
            crop_region = determine_crop_region(
                kps_scores, 
                org_height, 
                org_width
            )
        return kps_edges
