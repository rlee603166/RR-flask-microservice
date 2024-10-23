from flask import jsonify, request
import tensorflow as tf
import os

from . import api_bp
from app.services.data_service import DataProcessor
from app.services.pose_service import PoseDetection

data_processor = DataProcessor()
pose_detector = PoseDetection()

@api_bp.route('/')
def api_home():
    return jsonify({'message': 'Welcome to api landing page'})

@api_bp.route('/predict', methods=['POST'])
def predict_data():
    try:
        if 'gifs' not in request.files:
            return jsonify({'error': 'No gifs found'}), 400
        
        videos = request.files.getlist('gifs')
        pair_id = request.form.get('pair_id')
        
        front_impact_time = request.form.get('front_impact_time')
        back_impact_time = request.form.get('back_impact_time')
        
        if len(videos) < 2:
            return jsonify({'error': 'Submit 2 gifs!'}), 400
        
        front_temp = f"/tmp/{videos[0].filename}"
        back_temp = f"/tmp/{videos[1].filename}"
        
        videos[0].save(front_temp)
        videos[1].save(back_temp)
        
        front = tf.io.read_file(front_temp)
        back = tf.io.read_file(back_temp)
        
        front = tf.image.decode_gif(front)
        back = tf.image.decode_gif(back)
        
        front = pose_detector.predict(front)
        back = pose_detector.predict(back)
        
        front, back, _ = data_processor.post_process(front, front_impact_time, back, back_impact_time)
        
        os.remove(front_temp)
        os.remove(back_temp)
        
        return {
            'pair_id': pair_id,
            'front_kps': front,
            'back_kps': back
        }
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500