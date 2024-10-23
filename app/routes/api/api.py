from flask import jsonify
from . import api_bp

@api_bp.route('/')
def api_home():
    return jsonify({'message': 'Welcome to api landing page'})

@api_bp.route('/process', methods=['POST'])
def process_data():
    try:
        return jsonify({'result': 'hello'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500