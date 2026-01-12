"""
Example Usage - Programmatic API

This script demonstrates how to use the modules programmatically
instead of using the command-line interface.
"""

import cv2
import mediapipe as mp
from modules import (FaceRecognitionModule, ExerciseClassifier, 
                     SquatRepCounter, PushUpRepCounter)


def example_face_enrollment():
    """Example: Enroll a user for face recognition."""
    print("=== Face Enrollment Example ===")
    
    # Initialize face recognition module
    face_module = FaceRecognitionModule()
    
    # Enroll user from directory
    success = face_module.enroll_user_from_directory(
        name="John Doe",
        directory_path="data/enrollment/john"
    )
    
    if success:
        print(f"✓ Enrolled user successfully")
        print(f"Total users: {len(face_module.get_enrolled_users())}")
    else:
        print("✗ Enrollment failed")


def example_face_identification():
    """Example: Identify person in an image."""
    print("\n=== Face Identification Example ===")
    
    # Initialize face recognition module
    face_module = FaceRecognitionModule()
    
    # Load test image
    image = cv2.imread("data/videos/frame.jpg")
    
    if image is not None:
        # Identify person
        name, confidence = face_module.identify_person(image)
        print(f"Identified: {name} (confidence: {confidence:.3f})")
    else:
        print("Image not found - skipping example")


def example_exercise_classification():
    """Example: Classify exercise from video frame."""
    print("\n=== Exercise Classification Example ===")
    
    # Initialize pose and classifier
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    classifier = ExerciseClassifier()
    
    # Load test frame
    frame = cv2.imread("data/videos/frame.jpg")
    
    if frame is not None:
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Classify exercise
            exercise, confidence = classifier.classify_exercise(results.pose_landmarks)
            print(f"Exercise: {exercise} (confidence: {confidence:.3f})")
        else:
            print("No pose detected in frame")
    else:
        print("Image not found - skipping example")
    
    pose.close()


def example_rep_counting():
    """Example: Count reps in a video."""
    print("\n=== Rep Counting Example ===")
    
    # Initialize components
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    counter = SquatRepCounter()
    
    # Open video
    cap = cv2.VideoCapture("data/videos/squats.mp4")
    
    if not cap.isOpened():
        print("Video not found - skipping example")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Process pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Update counter
            counter.process_frame(results.pose_landmarks, timestamp)
    
    # Print results
    print(f"Total reps counted: {counter.get_rep_count()}")
    
    rep_details = counter.get_rep_details()
    for rep in rep_details[:3]:  # Show first 3 reps
        print(f"  Rep {rep['rep_num']}: {rep['duration']:.2f}s")
    
    cap.release()
    pose.close()


def example_custom_rep_counter():
    """Example: Create custom rep counter with adjusted thresholds."""
    print("\n=== Custom Rep Counter Example ===")
    
    # Create squat counter with stricter thresholds
    strict_counter = SquatRepCounter(
        down_threshold=80,   # Deeper squat required (default: 90)
        up_threshold=150     # More extended standing (default: 140)
    )
    
    print("Created strict squat counter:")
    print(f"  Down threshold: {strict_counter.down_threshold}°")
    print(f"  Up threshold: {strict_counter.up_threshold}°")
    
    # Create push-up counter
    pushup_counter = PushUpRepCounter()
    print("\nCreated push-up counter:")
    print(f"  Down threshold: {pushup_counter.down_threshold}°")
    print(f"  Up threshold: {pushup_counter.up_threshold}°")


def main():
    """Run all examples."""
    print("Workout Video Analysis - Example Usage\n")
    print("=" * 60)
    
    # Note: These examples will only work if you have:
    # 1. Enrolled users in the face recognition system
    # 2. Test images/videos in the data directories
    
    try:
        example_face_enrollment()
    except Exception as e:
        print(f"Face enrollment example failed: {e}")
    
    try:
        example_face_identification()
    except Exception as e:
        print(f"Face identification example failed: {e}")
    
    try:
        example_exercise_classification()
    except Exception as e:
        print(f"Exercise classification example failed: {e}")
    
    try:
        example_rep_counting()
    except Exception as e:
        print(f"Rep counting example failed: {e}")
    
    # This example always works (no external data needed)
    example_custom_rep_counter()
    
    print("\n" + "=" * 60)
    print("\nFor production use, see main.py and enroll_user.py")


if __name__ == '__main__':
    main()
