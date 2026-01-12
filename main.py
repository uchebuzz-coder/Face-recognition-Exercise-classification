"""
Main Video Analysis Pipeline

Process workout videos to identify people, classify exercises, and count reps.
"""

import argparse
import os
import sys
import time
import json
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional

from modules.face_recognition_module import FaceRecognitionModule
from modules.exercise_classifier import ExerciseClassifier
from modules.rep_counter import (SquatRepCounter, PushUpRepCounter, 
                                  LungeRepCounter, BicepCurlRepCounter)


class WorkoutVideoAnalyzer:
    """
    Main pipeline for analyzing workout videos.
    """
    
    def __init__(self, 
                 video_path: str,
                 output_path: str,
                 draw_overlays: bool = True,
                 face_check_interval: int = 30):
        """
        Initialize video analyzer.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            draw_overlays: Whether to draw overlays on video
            face_check_interval: Check face recognition every N frames
        """
        self.video_path = video_path
        self.output_path = output_path
        self.draw_overlays = draw_overlays
        self.face_check_interval = face_check_interval
        
        # Initialize modules
        self.face_module = FaceRecognitionModule()
        self.exercise_classifier = ExerciseClassifier()
        
        # Initialize rep counters for all exercises
        self.rep_counters = {
            'Squats': SquatRepCounter(),
            'Push-ups': PushUpRepCounter(),
            'Lunges': LungeRepCounter(),
            'Bicep Curls': BicepCurlRepCounter(),
        }
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis results
        self.identified_person = 'Unknown'
        self.person_confidence = 0.0
        self.current_exercise = 'Unknown'
        self.exercise_confidence = 0.0
        self.current_rep_counter = None
        
    def process_video(self) -> Dict:
        """
        Process the video and generate analysis results.
        
        Returns:
            Dictionary containing analysis results
        """
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Total Frames: {total_frames}")
        
        # Initialize video writer
        if self.draw_overlays:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        start_time = time.time()
        
        print("\nProcessing video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Process frame
            annotated_frame = self.process_frame(frame, frame_count, timestamp)
            
            # Write frame
            if self.draw_overlays and out is not None:
                out.write(annotated_frame)
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
        
        print(f"\n  Completed: {frame_count} frames processed")
        
        # Release resources
        cap.release()
        if self.draw_overlays:
            out.release()
        self.pose.close()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate results
        results = self.generate_results(duration, processing_time)
        
        return results
    
    def process_frame(self, frame, frame_count: int, timestamp: float):
        """
        Process a single frame.
        
        Args:
            frame: Video frame
            frame_count: Current frame number
            timestamp: Current timestamp in seconds
            
        Returns:
            Annotated frame
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb_frame)
        
        # Face recognition (every N frames)
        if frame_count % self.face_check_interval == 0:
            person, confidence = self.face_module.identify_person(frame)
            if confidence > 0:
                self.identified_person = person
                self.person_confidence = confidence
        
        # Exercise classification and rep counting
        if results.pose_landmarks:
            # Classify exercise
            exercise, confidence = self.exercise_classifier.classify_exercise(results.pose_landmarks)
            
            if confidence > 0.5:
                self.current_exercise = exercise
                self.exercise_confidence = confidence
                
                # Update rep counter for current exercise
                if exercise in self.rep_counters:
                    self.current_rep_counter = self.rep_counters[exercise]
                    self.current_rep_counter.process_frame(results.pose_landmarks, timestamp)
        
        # Draw overlays
        if self.draw_overlays:
            annotated_frame = self.draw_annotations(frame.copy(), results)
        else:
            annotated_frame = frame
        
        return annotated_frame
    
    def draw_annotations(self, frame, pose_results):
        """
        Draw annotations on frame.
        
        Args:
            frame: Video frame
            pose_results: MediaPipe pose results
            
        Returns:
            Annotated frame
        """
        height, width = frame.shape[:2]
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw semi-transparent overlay bars
        overlay = frame.copy()
        
        # Top bar for person and exercise info
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # Bottom bar for rep count
        cv2.rectangle(overlay, (0, height - 60), (width, height), (0, 0, 0), -1)
        
        # Blend overlay
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw person name (top-left)
        person_text = f"Person: {self.identified_person}"
        if self.person_confidence > 0:
            person_text += f" ({self.person_confidence:.2f})"
        cv2.putText(frame, person_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw exercise name (top-right)
        exercise_text = f"Exercise: {self.current_exercise}"
        if self.exercise_confidence > 0:
            exercise_text += f" ({self.exercise_confidence:.2f})"
        text_size = cv2.getTextSize(exercise_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(frame, exercise_text, (width - text_size[0] - 20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw rep count (bottom-center)
        if self.current_rep_counter:
            rep_count = self.current_rep_counter.get_rep_count()
            rep_text = f"Reps: {rep_count}"
            
            # Add last rep info if available
            rep_details = self.current_rep_counter.get_rep_details()
            if len(rep_details) > 0:
                last_rep = rep_details[-1]
                rep_text += f" | Last: {last_rep['duration']:.1f}s"
            
            text_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(frame, rep_text, (text_x, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return frame
    
    def generate_results(self, duration: float, processing_time: float) -> Dict:
        """
        Generate analysis results.
        
        Args:
            duration: Video duration in seconds
            processing_time: Processing time in seconds
            
        Returns:
            Dictionary of results
        """
        # Get rep details from the appropriate counter
        rep_details = []
        rep_count = 0
        
        if self.current_rep_counter:
            rep_details = self.current_rep_counter.get_rep_details()
            rep_count = self.current_rep_counter.get_rep_count()
        
        results = {
            'video_filename': os.path.basename(self.video_path),
            'duration_seconds': round(duration, 2),
            'identified_person': self.identified_person,
            'person_confidence': round(self.person_confidence, 3),
            'exercise_detected': self.current_exercise,
            'exercise_confidence': round(self.exercise_confidence, 3),
            'total_reps': rep_count,
            'reps_detail': rep_details,
            'processing_time_seconds': round(processing_time, 2)
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Analyze workout video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to output video (default: outputs/annotated_<input>.mp4)')
    parser.add_argument('--json', type=str, help='Path to JSON results file (default: outputs/<input>.json)')
    parser.add_argument('--no-video', action='store_true', help='Skip video output, only generate JSON')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Set default output paths
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    
    if args.output is None:
        args.output = f"outputs/annotated_{video_name}.mp4"
    
    if args.json is None:
        args.json = f"outputs/{video_name}_results.json"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    
    # Process video
    print(f"Analyzing video: {args.video}")
    
    analyzer = WorkoutVideoAnalyzer(
        video_path=args.video,
        output_path=args.output,
        draw_overlays=not args.no_video
    )
    
    results = analyzer.process_video()
    
    # Save JSON results
    with open(args.json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Person: {results['identified_person']} (confidence: {results['person_confidence']:.2f})")
    print(f"Exercise: {results['exercise_detected']} (confidence: {results['exercise_confidence']:.2f})")
    print(f"Total Reps: {results['total_reps']}")
    print(f"Processing Time: {results['processing_time_seconds']:.2f}s")
    print(f"\nResults saved to: {args.json}")
    
    if not args.no_video:
        print(f"Annotated video saved to: {args.output}")
    
    print("="*60)


if __name__ == '__main__':
    main()
