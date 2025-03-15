from flask import jsonify, request
import tensorflow as tf
import time
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
        if 'file' not in request.files:
            return jsonify({'error': 'No files found'}), 400
        
        videos = request.files.getlist('file')
        user_id = request.form.get('user_id', 'anonymous')
        
        front_impact_time = float(request.form.get('front_impact_time', 0.5))
        back_impact_time = float(request.form.get('back_impact_time', 0.5))
        
        # Get durations from frontend
        front_duration = float(request.form.get('front_duration', 0))
        back_duration = float(request.form.get('back_duration', 0))
        
        if len(videos) < 2:
            return jsonify({'error': 'Submit both front and back videos!'}), 400
        
        # Create a unique process ID
        process_id = f"{user_id}_{int(time.time())}"
        
        # Convert videos to GIFs for processing
        front_gif_path = f"/tmp/front_{process_id}.gif"
        back_gif_path = f"/tmp/back_{process_id}.gif"
        
        # Save the uploaded videos temporarily
        front_temp = f"/tmp/front_{process_id}.mp4"
        back_temp = f"/tmp/back_{process_id}.mp4"
        
        videos[0].save(front_temp)
        videos[1].save(back_temp)
        
        # Convert videos to GIFs using ffmpeg, now using durations from frontend
        convert_to_gif_with_duration(front_temp, front_gif_path, front_impact_time, front_duration)
        convert_to_gif_with_duration(back_temp, back_gif_path, back_impact_time, back_duration)
        
       
        front = tf.io.read_file(front_gif_path)  # Use the GIF path
        back = tf.io.read_file(back_gif_path)    # Use the GIF path

        front = tf.image.decode_gif(front)
        back = tf.image.decode_gif(back)       
        front = pose_detector.predict(front)
        back = pose_detector.predict(back)

        result = data_processor.post_process(front, front_impact_time, back, back_impact_time)
        front, back = result[0], result[1]

        os.remove(front_temp)
        os.remove(back_temp)
        
        return {
            'front_kps': front,
            'back_kps': back
        }
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_to_gif_with_duration(video_path, gif_path, impact_time, duration):
    """Convert video to GIF centered around the impact time using duration from frontend"""
    import subprocess
    
    impact_second = duration * impact_time
    start_time = max(0, impact_second - 1)
    
    subprocess.call([
        'ffmpeg', '-y', '-i', video_path, 
        '-ss', str(start_time), '-t', '2', 
        '-vf', 'fps=10,scale=320:-1', 
        gif_path
    ])
