from flask import Blueprint, jsonify

landing_bp = Blueprint('landing', __name__)

from . import landing