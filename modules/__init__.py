"""
Workout Video Analysis Modules
"""

from .face_recognition_module import FaceRecognitionModule
from .exercise_classifier import ExerciseClassifier
from .rep_counter import (SquatRepCounter, PushUpRepCounter, 
                          LungeRepCounter, BicepCurlRepCounter)
from .utils import calculate_angle, calculate_distance, get_landmark_coords

__all__ = [
    'FaceRecognitionModule',
    'ExerciseClassifier',
    'SquatRepCounter',
    'PushUpRepCounter',
    'LungeRepCounter',
    'BicepCurlRepCounter',
    'calculate_angle',
    'calculate_distance',
    'get_landmark_coords'
]
