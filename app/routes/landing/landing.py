from flask import jsonify
from . import landing_bp

@landing_bp.route('/')
def hello():
    return jsonify({'message': 'Hello World!'}), 200

@landing_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

