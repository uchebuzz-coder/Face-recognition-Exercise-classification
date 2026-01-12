"""
Rep Counting Module

Counts exercise repetitions using pose-based state machines.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Tuple, Optional
import mediapipe as mp

from .utils import calculate_angle, get_landmark_coords


class RepState(Enum):
    """States for rep counting state machine."""
    STANDING = "standing"
    DOWN = "down"
    TRANSITION_UP = "transition_up"
    TRANSITION_DOWN = "transition_down"


class RepCounter(ABC):
    """
    Abstract base class for rep counters.
    """
    
    def __init__(self):
        self.rep_count = 0
        self.reps_detail = []
        self.current_state = RepState.STANDING
        self.current_rep_start = None
        
    @abstractmethod
    def process_frame(self, landmarks, timestamp: float) -> None:
        """
        Process a frame and update state.
        
        Args:
            landmarks: MediaPipe pose landmarks
            timestamp: Current timestamp in seconds
        """
        pass
    
    def get_rep_count(self) -> int:
        """Get current rep count."""
        return self.rep_count
    
    def get_rep_details(self) -> List[Dict]:
        """Get detailed rep information."""
        return self.reps_detail
    
    def reset(self) -> None:
        """Reset counter."""
        self.rep_count = 0
        self.reps_detail = []
        self.current_state = RepState.STANDING
        self.current_rep_start = None


class SquatRepCounter(RepCounter):
    """
    Rep counter for squat exercise using hip and knee angles.
    """
    
    def __init__(self, down_threshold: float = 90, up_threshold: float = 140):
        """
        Initialize squat rep counter.
        
        Args:
            down_threshold: Hip angle threshold for down position (degrees)
            up_threshold: Hip angle threshold for up position (degrees)
        """
        super().__init__()
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.mp_pose = mp.solutions.pose
        self.bottom_time = None
        
    def process_frame(self, landmarks, timestamp: float) -> None:
        """
        Process frame and update squat rep count.
        
        Args:
            landmarks: MediaPipe pose landmarks
            timestamp: Current timestamp in seconds
        """
        if landmarks is None:
            return
        
        # Get key landmarks
        left_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        
        # Calculate hip angles (key metric for squats)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # State machine logic
        if self.current_state == RepState.STANDING:
            # Check if going down
            if avg_hip_angle < self.down_threshold:
                self.current_state = RepState.DOWN
                self.current_rep_start = timestamp
                self.bottom_time = timestamp
                
        elif self.current_state == RepState.DOWN:
            # Update bottom time
            self.bottom_time = timestamp
            
            # Check if coming back up
            if avg_hip_angle > self.up_threshold:
                # Complete rep
                self.rep_count += 1
                
                rep_info = {
                    'rep_num': self.rep_count,
                    'start_time': round(self.current_rep_start, 2),
                    'bottom_time': round(self.bottom_time, 2),
                    'end_time': round(timestamp, 2),
                    'duration': round(timestamp - self.current_rep_start, 2)
                }
                self.reps_detail.append(rep_info)
                
                # Reset state
                self.current_state = RepState.STANDING
                self.current_rep_start = None
                self.bottom_time = None


class PushUpRepCounter(RepCounter):
    """
    Rep counter for push-up exercise using elbow angles.
    """
    
    def __init__(self, down_threshold: float = 90, up_threshold: float = 140):
        """
        Initialize push-up rep counter.
        
        Args:
            down_threshold: Elbow angle threshold for down position (degrees)
            up_threshold: Elbow angle threshold for up position (degrees)
        """
        super().__init__()
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.mp_pose = mp.solutions.pose
        self.bottom_time = None
        
    def process_frame(self, landmarks, timestamp: float) -> None:
        """
        Process frame and update push-up rep count.
        
        Args:
            landmarks: MediaPipe pose landmarks
            timestamp: Current timestamp in seconds
        """
        if landmarks is None:
            return
        
        # Get key landmarks
        left_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        
        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # State machine logic
        if self.current_state == RepState.STANDING:  # "Up" position for push-ups
            # Check if going down
            if avg_elbow_angle < self.down_threshold:
                self.current_state = RepState.DOWN
                self.current_rep_start = timestamp
                self.bottom_time = timestamp
                
        elif self.current_state == RepState.DOWN:
            # Update bottom time
            self.bottom_time = timestamp
            
            # Check if coming back up
            if avg_elbow_angle > self.up_threshold:
                # Complete rep
                self.rep_count += 1
                
                rep_info = {
                    'rep_num': self.rep_count,
                    'start_time': round(self.current_rep_start, 2),
                    'bottom_time': round(self.bottom_time, 2),
                    'end_time': round(timestamp, 2),
                    'duration': round(timestamp - self.current_rep_start, 2)
                }
                self.reps_detail.append(rep_info)
                
                # Reset state
                self.current_state = RepState.STANDING
                self.current_rep_start = None
                self.bottom_time = None


class LungeRepCounter(RepCounter):
    """
    Rep counter for lunge exercise using knee angles.
    """
    
    def __init__(self, down_threshold: float = 100, up_threshold: float = 150):
        """
        Initialize lunge rep counter.
        
        Args:
            down_threshold: Knee angle threshold for down position (degrees)
            up_threshold: Knee angle threshold for up position (degrees)
        """
        super().__init__()
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.mp_pose = mp.solutions.pose
        self.bottom_time = None
        
    def process_frame(self, landmarks, timestamp: float) -> None:
        """
        Process frame and update lunge rep count.
        
        Args:
            landmarks: MediaPipe pose landmarks
            timestamp: Current timestamp in seconds
        """
        if landmarks is None:
            return
        
        # Get key landmarks
        left_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE)
        left_ankle = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Use minimum angle (the bent leg)
        min_knee_angle = min(left_knee_angle, right_knee_angle)
        
        # State machine logic
        if self.current_state == RepState.STANDING:
            # Check if going down
            if min_knee_angle < self.down_threshold:
                self.current_state = RepState.DOWN
                self.current_rep_start = timestamp
                self.bottom_time = timestamp
                
        elif self.current_state == RepState.DOWN:
            # Update bottom time
            self.bottom_time = timestamp
            
            # Check if coming back up
            if min_knee_angle > self.up_threshold:
                # Complete rep
                self.rep_count += 1
                
                rep_info = {
                    'rep_num': self.rep_count,
                    'start_time': round(self.current_rep_start, 2),
                    'bottom_time': round(self.bottom_time, 2),
                    'end_time': round(timestamp, 2),
                    'duration': round(timestamp - self.current_rep_start, 2)
                }
                self.reps_detail.append(rep_info)
                
                # Reset state
                self.current_state = RepState.STANDING
                self.current_rep_start = None
                self.bottom_time = None


class BicepCurlRepCounter(RepCounter):
    """
    Rep counter for bicep curl exercise using elbow angles.
    """
    
    def __init__(self, curl_threshold: float = 50, extend_threshold: float = 140):
        """
        Initialize bicep curl rep counter.
        
        Args:
            curl_threshold: Elbow angle threshold for curled position (degrees)
            extend_threshold: Elbow angle threshold for extended position (degrees)
        """
        super().__init__()
        self.curl_threshold = curl_threshold
        self.extend_threshold = extend_threshold
        self.mp_pose = mp.solutions.pose
        self.curl_time = None
        
    def process_frame(self, landmarks, timestamp: float) -> None:
        """
        Process frame and update bicep curl rep count.
        
        Args:
            landmarks: MediaPipe pose landmarks
            timestamp: Current timestamp in seconds
        """
        if landmarks is None:
            return
        
        # Get key landmarks
        left_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        
        # Calculate elbow angles
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        
        # State machine logic (extended -> curled -> extended)
        if self.current_state == RepState.STANDING:  # Extended position
            # Check if curling
            if avg_elbow_angle < self.curl_threshold:
                self.current_state = RepState.DOWN  # "Down" means curled for bicep curls
                self.current_rep_start = timestamp
                self.curl_time = timestamp
                
        elif self.current_state == RepState.DOWN:  # Curled position
            # Update curl time
            self.curl_time = timestamp
            
            # Check if extending
            if avg_elbow_angle > self.extend_threshold:
                # Complete rep
                self.rep_count += 1
                
                rep_info = {
                    'rep_num': self.rep_count,
                    'start_time': round(self.current_rep_start, 2),
                    'curl_time': round(self.curl_time, 2),
                    'end_time': round(timestamp, 2),
                    'duration': round(timestamp - self.current_rep_start, 2)
                }
                self.reps_detail.append(rep_info)
                
                # Reset state
                self.current_state = RepState.STANDING
                self.current_rep_start = None
                self.curl_time = None
