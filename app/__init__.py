from flask import Flask
from flask_cors import CORS

from .routes.landing import landing_bp
from .routes.api import api_bp

def create_app():
    app = Flask(__name__)
    
    CORS(app)
    
    app.register_blueprint(landing_bp)
    app.register_blueprint(api_bp)
    
    return app