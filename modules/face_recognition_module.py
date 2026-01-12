"""
Face Recognition Module

Identifies known individuals using facial embeddings with InsightFace.
"""

import os
import pickle
import numpy as np
from typing import Tuple, List, Dict, Optional
import cv2

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


class FaceRecognitionModule:
    """
    Face recognition using InsightFace with ArcFace embeddings.
    """
    
    def __init__(self, embeddings_path: str = 'data/embeddings.pkl', 
                 similarity_threshold: float = 0.6):
        """
        Initialize face recognition module.
        
        Args:
            embeddings_path: Path to save/load face embeddings
            similarity_threshold: Minimum cosine similarity for match
        """
        self.embeddings_path = embeddings_path
        self.similarity_threshold = similarity_threshold
        self.known_embeddings = {}  # {name: [embedding1, embedding2, ...]}
        
        # Initialize InsightFace
        if FaceAnalysis is None:
            print("Warning: InsightFace not installed. Face recognition will be disabled.")
            self.face_app = None
        else:
            try:
                self.face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception as e:
                print(f"Warning: Could not initialize InsightFace: {e}")
                self.face_app = None
        
        # Load existing embeddings
        self.load_embeddings()
    
    def load_embeddings(self) -> None:
        """Load face embeddings from disk."""
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                print(f"Loaded embeddings for {len(self.known_embeddings)} users")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                self.known_embeddings = {}
        else:
            print("No existing embeddings found")
    
    def save_embeddings(self) -> None:
        """Save face embeddings to disk."""
        try:
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
            print(f"Saved embeddings for {len(self.known_embeddings)} users")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    def extract_embedding(self, image) -> Optional[np.ndarray]:
        """
        Extract face embedding from image.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            512-dim embedding vector or None if no face detected
        """
        if self.face_app is None:
            return None
        
        try:
            # Detect faces and extract embeddings
            faces = self.face_app.get(image)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face (by bounding box area)
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            return largest_face.embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def enroll_user(self, name: str, image_paths: List[str]) -> bool:
        """
        Enroll a new user with multiple images.
        
        Args:
            name: User's name
            image_paths: List of paths to user's images
            
        Returns:
            True if successful, False otherwise
        """
        if self.face_app is None:
            print("Face recognition not available")
            return False
        
        embeddings = []
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image: {image_path}")
                continue
            
            # Extract embedding
            embedding = self.extract_embedding(image)
            
            if embedding is not None:
                embeddings.append(embedding)
                print(f"Extracted embedding from {os.path.basename(image_path)}")
            else:
                print(f"Warning: No face detected in {image_path}")
        
        if len(embeddings) == 0:
            print(f"Error: No valid embeddings extracted for {name}")
            return False
        
        # Store embeddings
        self.known_embeddings[name] = embeddings
        self.save_embeddings()
        
        print(f"Enrolled {name} with {len(embeddings)} embeddings")
        return True
    
    def enroll_user_from_directory(self, name: str, directory_path: str) -> bool:
        """
        Enroll a user from all images in a directory.
        
        Args:
            name: User's name
            directory_path: Path to directory containing user's images
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(directory_path):
            print(f"Error: Directory not found: {directory_path}")
            return False
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for filename in os.listdir(directory_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(directory_path, filename))
        
        if len(image_paths) == 0:
            print(f"Error: No images found in {directory_path}")
            return False
        
        return self.enroll_user(name, image_paths)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def identify_person(self, frame) -> Tuple[str, float]:
        """
        Identify person in frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            Tuple of (name, confidence) or ('Unknown', 0.0) if no match
        """
        if self.face_app is None:
            return ('Unknown', 0.0)
        
        if len(self.known_embeddings) == 0:
            return ('Unknown', 0.0)
        
        # Extract embedding from frame
        embedding = self.extract_embedding(frame)
        
        if embedding is None:
            return ('Unknown', 0.0)
        
        # Compare with all known embeddings
        best_match = None
        best_similarity = 0.0
        
        for name, known_embs in self.known_embeddings.items():
            for known_emb in known_embs:
                similarity = self.cosine_similarity(embedding, known_emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
        
        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            return (best_match, best_similarity)
        else:
            return ('Unknown', 0.0)
    
    def get_enrolled_users(self) -> List[str]:
        """Get list of enrolled user names."""
        return list(self.known_embeddings.keys())
