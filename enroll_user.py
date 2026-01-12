"""
User Enrollment Script

Script to enroll users for face recognition.
"""

import argparse
import os
import sys

from modules.face_recognition_module import FaceRecognitionModule


def main():
    parser = argparse.ArgumentParser(description='Enroll a user for face recognition')
    parser.add_argument('--name', type=str, required=True, help='User name')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to user images')
    parser.add_argument('--directory', type=str, help='Directory containing user images')
    
    args = parser.parse_args()
    
    # Initialize face recognition module
    face_module = FaceRecognitionModule()
    
    # Enroll user
    if args.directory:
        success = face_module.enroll_user_from_directory(args.name, args.directory)
    elif args.images:
        success = face_module.enroll_user(args.name, args.images)
    else:
        print("Error: Must provide either --images or --directory")
        sys.exit(1)
    
    if success:
        print(f"\n✓ Successfully enrolled {args.name}")
        print(f"Total enrolled users: {len(face_module.get_enrolled_users())}")
        print(f"Users: {', '.join(face_module.get_enrolled_users())}")
    else:
        print(f"\n✗ Failed to enroll {args.name}")
        sys.exit(1)


if __name__ == '__main__':
    main()
