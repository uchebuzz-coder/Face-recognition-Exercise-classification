"""
Utility functions for pose analysis
"""

import numpy as np
from typing import Tuple, List


def calculate_angle(point1: Tuple[float, float, float],
                   point2: Tuple[float, float, float],
                   point3: Tuple[float, float, float]) -> float:
    """
    Calculate angle between three points (in degrees).
    
    Args:
        point1: First point (x, y, z)
        point2: Vertex point (x, y, z)
        point3: Third point (x, y, z)
    
    Returns:
        Angle in degrees
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def calculate_distance(point1: Tuple[float, float, float],
                      point2: Tuple[float, float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y, z)
        point2: Second point (x, y, z)
    
    Returns:
        Distance
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.linalg.norm(p1 - p2)


def get_landmark_coords(landmarks, landmark_id: int) -> Tuple[float, float, float]:
    """
    Extract (x, y, z) coordinates from MediaPipe landmark.
    
    Args:
        landmarks: MediaPipe pose landmarks
        landmark_id: ID of the landmark to extract
    
    Returns:
        Tuple of (x, y, z) coordinates
    """
    landmark = landmarks.landmark[landmark_id]
    return (landmark.x, landmark.y, landmark.z)


def smooth_signal(values: List[float], window_size: int = 5) -> float:
    """
    Apply moving average smoothing to a signal.
    
    Args:
        values: List of values to smooth
        window_size: Size of smoothing window
    
    Returns:
        Smoothed value (mean of recent values)
    """
    if len(values) == 0:
        return 0.0
    
    recent_values = values[-window_size:]
    return np.mean(recent_values)
