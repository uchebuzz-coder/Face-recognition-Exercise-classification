"""
Exercise Classification Module

Classifies exercises based on pose landmarks using rule-based approach.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import deque
import mediapipe as mp

from .utils import calculate_angle, get_landmark_coords


class ExerciseClassifier:
    """
    Rule-based exercise classifier using pose keypoints.
    Supports: Squats, Push-ups, Lunges, Bicep Curls
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize exercise classifier.
        
        Args:
            window_size: Number of frames for temporal smoothing
        """
        self.window_size = window_size
        self.pose_buffer = deque(maxlen=window_size)
        self.mp_pose = mp.solutions.pose
        
        # Exercise names
        self.exercises = ['Squats', 'Push-ups', 'Lunges', 'Bicep Curls', 'Unknown']
        
    def extract_pose_features(self, landmarks) -> Dict[str, float]:
        """
        Extract relevant features from pose landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of feature values
        """
        if landmarks is None:
            return {}
        
        # Get key landmark coordinates
        left_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # Calculate angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Calculate body orientation (vertical vs horizontal)
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        torso_vertical = abs(avg_shoulder_y - avg_hip_y)
        
        # Calculate leg asymmetry (for lunges)
        knee_y_diff = abs(left_knee[1] - right_knee[1])
        ankle_x_diff = abs(left_ankle[0] - right_ankle[0])
        
        features = {
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'avg_elbow_angle': (left_elbow_angle + right_elbow_angle) / 2,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'avg_knee_angle': (left_knee_angle + right_knee_angle) / 2,
            'left_hip_angle': left_hip_angle,
            'right_hip_angle': right_hip_angle,
            'avg_hip_angle': (left_hip_angle + right_hip_angle) / 2,
            'torso_vertical': torso_vertical,
            'knee_y_diff': knee_y_diff,
            'ankle_x_diff': ankle_x_diff,
            'shoulder_y': avg_shoulder_y,
            'wrist_avg_y': (left_wrist[1] + right_wrist[1]) / 2,
        }
        
        return features
    
    def classify_squat(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for squat exercise."""
        confidence = 0.0
        
        # Check hip flexion (key indicator)
        if features['avg_hip_angle'] < 120:
            confidence += 0.4
        
        # Check knee flexion
        if features['avg_knee_angle'] < 140:
            confidence += 0.3
        
        # Check upright torso (vertical orientation)
        if features['torso_vertical'] > 0.15:
            confidence += 0.2
        
        # Check symmetric leg position
        if features['knee_y_diff'] < 0.1:
            confidence += 0.1
        
        return confidence
    
    def classify_pushup(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for push-up exercise."""
        confidence = 0.0
        
        # Check elbow flexion
        if features['avg_elbow_angle'] < 140:
            confidence += 0.4
        
        # Check horizontal body orientation (low torso vertical)
        if features['torso_vertical'] < 0.15:
            confidence += 0.3
        
        # Check if shoulders are at similar height as hips (plank position)
        if abs(features['shoulder_y'] - 0.5) < 0.2:  # Normalized coordinates
            confidence += 0.2
        
        # Check wrists are in front (supporting position)
        if features['wrist_avg_y'] > 0.3:
            confidence += 0.1
        
        return confidence
    
    def classify_lunge(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for lunge exercise."""
        confidence = 0.0
        
        # Check asymmetric leg position (key indicator)
        if features['knee_y_diff'] > 0.1:
            confidence += 0.4
        
        # Check forward stance (ankle separation)
        if features['ankle_x_diff'] > 0.2:
            confidence += 0.3
        
        # Check knee flexion on one leg
        min_knee_angle = min(features['left_knee_angle'], features['right_knee_angle'])
        if min_knee_angle < 120:
            confidence += 0.2
        
        # Check upright torso
        if features['torso_vertical'] > 0.15:
            confidence += 0.1
        
        return confidence
    
    def classify_bicep_curl(self, features: Dict[str, float]) -> float:
        """Calculate confidence score for bicep curl exercise."""
        confidence = 0.0
        
        # Check elbow flexion
        if features['avg_elbow_angle'] < 100:
            confidence += 0.4
        
        # Check upright torso
        if features['torso_vertical'] > 0.15:
            confidence += 0.3
        
        # Check wrists elevated (curling motion)
        if features['wrist_avg_y'] < features['shoulder_y']:
            confidence += 0.2
        
        # Check relatively extended legs (standing)
        if features['avg_knee_angle'] > 150:
            confidence += 0.1
        
        return confidence
    
    def classify_frame(self, landmarks) -> Tuple[str, float]:
        """
        Classify exercise in a single frame.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Tuple of (exercise_name, confidence)
        """
        if landmarks is None:
            return ('Unknown', 0.0)
        
        features = self.extract_pose_features(landmarks)
        
        if not features:
            return ('Unknown', 0.0)
        
        # Calculate confidence for each exercise
        scores = {
            'Squats': self.classify_squat(features),
            'Push-ups': self.classify_pushup(features),
            'Lunges': self.classify_lunge(features),
            'Bicep Curls': self.classify_bicep_curl(features),
        }
        
        # Get exercise with highest confidence
        best_exercise = max(scores.items(), key=lambda x: x[1])
        
        # Return Unknown if confidence too low
        if best_exercise[1] < 0.4:
            return ('Unknown', 0.0)
        
        return best_exercise
    
    def classify_exercise(self, landmarks) -> Tuple[str, float]:
        """
        Classify exercise with temporal smoothing.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Tuple of (exercise_name, confidence)
        """
        # Get current frame classification
        exercise, confidence = self.classify_frame(landmarks)
        
        # Add to buffer
        self.pose_buffer.append((exercise, confidence))
        
        # If buffer not full yet, return current classification
        if len(self.pose_buffer) < self.window_size // 2:
            return exercise, confidence
        
        # Perform majority voting on recent frames
        exercise_votes = {}
        confidence_sum = {}
        
        for ex, conf in self.pose_buffer:
            if ex not in exercise_votes:
                exercise_votes[ex] = 0
                confidence_sum[ex] = 0.0
            exercise_votes[ex] += 1
            confidence_sum[ex] += conf
        
        # Get exercise with most votes
        best_exercise = max(exercise_votes.items(), key=lambda x: x[1])
        avg_confidence = confidence_sum[best_exercise[0]] / best_exercise[1]
        
        return best_exercise[0], avg_confidence
    
    def reset(self):
        """Reset the classifier state."""
        self.pose_buffer.clear()
