# System Architecture

Detailed technical architecture of the Workout Video Analysis MVP.

## Overview

The system follows a modular, pipeline-based architecture with clear separation of concerns. Each module can be tested and extended independently.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Layer                             │
│  ┌──────────────┐           ┌────────────────┐              │
│  │ Video File   │           │ Enrollment     │              │
│  │ (.mp4, .avi) │           │ Images         │              │
│  └──────┬───────┘           └────────┬───────┘              │
└─────────┼────────────────────────────┼──────────────────────┘
          │                            │
          │                            │ Enrollment Phase
          │                            ▼
          │                   ┌─────────────────┐
          │                   │ Face Recognition│
          │                   │ Module          │
          │                   │ - Extract embed │
          │                   │ - Store in DB   │
          │                   └─────────────────┘
          │                            │
          │                            │ embeddings.pkl
          │                            ▼
          │                   ┌─────────────────┐
          │                   │ Embeddings DB   │
          │                   └─────────────────┘
          │
          │ Processing Phase
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Processing Layer                           │
│                                                               │
│  ┌────────────────────────────────────────────────┐          │
│  │            Frame Processing Loop               │          │
│  │                                                │          │
│  │  For each frame:                               │          │
│  │  1. Extract RGB                                │          │
│  │  2. Run MediaPipe Pose                         │          │
│  │  3. Face ID (every 30 frames)                  │          │
│  │  4. Exercise Classification                    │          │
│  │  5. Rep Counting                               │          │
│  │  6. Draw Overlays                              │          │
│  └───┬───────────────┬────────────────┬───────────┘          │
│      │               │                │                      │
│      ▼               ▼                ▼                      │
│  ┌────────┐   ┌─────────────┐  ┌──────────────┐            │
│  │ Face   │   │  Exercise   │  │ Rep Counter  │            │
│  │ Module │   │  Classifier │  │ (4 types)    │            │
│  └────────┘   └─────────────┘  └──────────────┘            │
│      │               │                │                      │
│      │ person        │ exercise       │ rep count            │
│      │ confidence    │ confidence     │ timestamps           │
│      │               │                │                      │
└──────┼───────────────┼────────────────┼──────────────────────┘
       │               │                │
       └───────┬───────┴────────┬───────┘
               │                │
               ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                              │
│                                                               │
│  ┌──────────────────────┐      ┌─────────────────────┐      │
│  │  Annotated Video     │      │  JSON Results       │      │
│  │  - Person name       │      │  - Person ID        │      │
│  │  - Exercise label    │      │  - Exercise type    │      │
│  │  - Rep count         │      │  - Rep details      │      │
│  │  - Pose skeleton     │      │  - Timestamps       │      │
│  └──────────────────────┘      └─────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Face Recognition Module

**File**: `modules/face_recognition_module.py`

**Technology**: InsightFace with ArcFace model

**Key Components**:
- `FaceAnalysis`: InsightFace API wrapper
- Embedding extraction: Converts face → 512-dim vector
- Database: Pickle file storing {name: [embeddings]}
- Similarity matching: Cosine similarity with threshold

**Data Flow**:
```
Image → Face Detection → Embedding Extraction → Cosine Similarity → (Name, Confidence)
```

**Performance**:
- Embedding extraction: ~20ms per face (GPU)
- Database lookup: O(n*m) where n=users, m=embeddings per user
- Typical: <50ms for 10 users with 5 embeddings each

### 2. Exercise Classification Module

**File**: `modules/exercise_classifier.py`

**Approach**: Rule-based classifier with temporal smoothing

**Pipeline**:
```
Landmarks → Feature Extraction → Rule Matching → Temporal Smoothing → (Exercise, Confidence)
```

**Features Extracted**:
- Joint angles: hip, knee, elbow (3 points → angle)
- Body orientation: torso verticality
- Leg asymmetry: position differences
- Relative positions: wrist height, shoulder position

**Classification Rules**:

| Exercise    | Primary Features | Thresholds |
|-------------|------------------|------------|
| Squats      | Hip angle < 120° | 0.4 + knee (0.3) + torso (0.2) |
| Push-ups    | Elbow < 140°, horizontal body | 0.4 + orientation (0.3) |
| Lunges      | Leg asymmetry > 0.1 | 0.4 + stance (0.3) |
| Bicep Curls | Elbow < 100°, upright | 0.4 + torso (0.3) |

**Temporal Smoothing**:
- Window: 30 frames (~1 second at 30fps)
- Method: Majority voting
- Reduces false positives from transient poses

### 3. Rep Counting Module

**File**: `modules/rep_counter.py`

**Approach**: Finite State Machine (FSM)

**State Machine (Squats)**:
```
     hip > 140°         hip < 90°          hip > 140°
STANDING ────────→  DOWN ────────→  STANDING
                                       ↓
                                    Count++
```

**Generic FSM Pattern**:
```python
class RepCounter:
    states = [STANDING, DOWN, TRANSITION_UP, TRANSITION_DOWN]
    
    def process_frame(landmarks, timestamp):
        angle = calculate_key_angle(landmarks)
        
        if current_state == STANDING and angle < down_threshold:
            current_state = DOWN
            rep_start_time = timestamp
        
        elif current_state == DOWN and angle > up_threshold:
            current_state = STANDING
            rep_count++
            save_rep_details(rep_start_time, timestamp)
```

**Rep Details Tracked**:
- `rep_num`: Sequential number
- `start_time`: When rep began
- `bottom_time`: Lowest point
- `end_time`: When rep completed
- `duration`: Total time for rep

### 4. Pose Estimation (MediaPipe)

**Technology**: MediaPipe Pose (Google)

**Landmarks**: 33 body keypoints
```
Key landmarks used:
- 11, 12: Shoulders
- 13, 14: Elbows
- 15, 16: Wrists
- 23, 24: Hips
- 25, 26: Knees
- 27, 28: Ankles
```

**Output**: For each landmark:
- x, y: Normalized coordinates [0, 1]
- z: Depth (relative to hips)
- visibility: Confidence score

**Performance**:
- Model complexity 1 (default): ~25ms per frame (GPU)
- Accuracy: >95% for visible body parts

### 5. Main Pipeline

**File**: `main.py`

**Processing Loop**:
```python
for each frame in video:
    # 1. Pose estimation (every frame)
    landmarks = mediapipe_pose.process(frame)
    
    # 2. Face recognition (every 30 frames)
    if frame_count % 30 == 0:
        person, confidence = face_module.identify(frame)
    
    # 3. Exercise classification (with temporal buffer)
    exercise, confidence = classifier.classify(landmarks)
    
    # 4. Rep counting (state machine update)
    if exercise in rep_counters:
        rep_counters[exercise].process_frame(landmarks, timestamp)
    
    # 5. Draw overlays
    annotated_frame = draw_overlays(frame, person, exercise, reps)
    
    # 6. Write output
    video_writer.write(annotated_frame)
```

**Optimization Strategies**:
1. **Sparse face recognition**: Check every 30 frames (assumes person doesn't change)
2. **Temporal buffering**: Smooth exercise classification over 30 frames
3. **Conditional rep counting**: Only count for detected exercise
4. **GPU acceleration**: MediaPipe and InsightFace use GPU when available

## Data Flow

### Enrollment Phase
```
Image Files → Face Detection → Embedding Extraction → Pickle File
                                                           ↓
                                               {name: [embeddings]}
```

### Video Processing Phase
```
Video File
    ↓
Frame Extraction
    ↓
    ├─→ MediaPipe Pose → Landmarks → Exercise Classifier → Exercise Label
    │                      ↓                                      ↓
    │                      └──────→ Rep Counter ────────→ Rep Count
    │                                                             ↓
    └─→ Face Recognition (sparse) ──────────────→ Person Name    ↓
                                                        ↓         ↓
                                    Overlay Generator ←┴─────────┘
                                            ↓
                                    Annotated Frame
                                            ↓
                                    Video Writer → Output Video
```

## Performance Characteristics

### Processing Speed

| Component | GPU (ms/frame) | CPU (ms/frame) |
|-----------|---------------|---------------|
| MediaPipe Pose | 25 | 80 |
| Face Recognition | 20* | 60* |
| Exercise Classifier | 2 | 2 |
| Rep Counter | <1 | <1 |
| Video I/O | 5 | 5 |
| **Total** | **52** | **147** |

*Per-frame cost amortized over 30 frames (actual: GPU=600ms/30, CPU=1800ms/30)

**Throughput**:
- GPU: ~19 FPS (52ms per frame)
- CPU: ~7 FPS (147ms per frame)

### Memory Usage

| Component | Memory |
|-----------|--------|
| MediaPipe Models | ~50 MB |
| InsightFace Models | ~100 MB |
| Face Embeddings DB | ~5 KB per user |
| Video Frame Buffer | ~6 MB (1080p) |
| **Total** | ~200 MB + video |

## Error Handling

### Graceful Degradation

1. **No GPU**: Falls back to CPU (slower but functional)
2. **InsightFace unavailable**: Face recognition disabled, rest works
3. **No face detected**: Returns "Unknown", continues processing
4. **No pose detected**: Skips frame, continues
5. **Exercise uncertain**: Returns "Unknown", doesn't count reps

### Validation

- **Input video**: Check file exists and is readable
- **Enrollment images**: Verify face detected before storing
- **Landmarks**: Check visibility scores before angle calculation
- **Rep counting**: Validate angle ranges before state transitions

## Extensibility

### Adding New Exercise

1. **Add classification rule** (`exercise_classifier.py`):
```python
def classify_new_exercise(self, features):
    confidence = 0.0
    if features['key_angle'] < threshold:
        confidence += 0.4
    return confidence
```

2. **Create rep counter** (`rep_counter.py`):
```python
class NewExerciseRepCounter(RepCounter):
    def process_frame(self, landmarks, timestamp):
        # Implement FSM
        pass
```

3. **Register in pipeline** (`main.py`):
```python
self.rep_counters['New Exercise'] = NewExerciseRepCounter()
```

### Tuning Parameters

**Face Recognition**:
- `similarity_threshold`: Lower = more lenient matching (default: 0.6)
- `face_check_interval`: More frequent = better tracking (default: 30)

**Exercise Classification**:
- Angle thresholds: Adjust per exercise in `classify_X` methods
- `window_size`: Larger = smoother but slower response (default: 30)

**Rep Counting**:
- `down_threshold`: Stricter = requires deeper motion (default: 90°)
- `up_threshold`: Stricter = requires fuller extension (default: 140°)

## Technology Stack Rationale

| Technology | Why Chosen | Alternatives |
|-----------|------------|-------------|
| MediaPipe | Fast, accurate, easy to use | OpenPose (slower), MMPose (complex) |
| InsightFace | SOTA accuracy, good docs | face_recognition (older), DeepFace (slower) |
| OpenCV | Industry standard, well-supported | Pillow (less video support) |
| Rule-based classifier | Simple, interpretable, no training | ML model (requires data, training) |
| FSM rep counter | Reliable, predictable | Heuristic counting (less accurate) |

## Future Improvements

### Short-term (< 1 week)
- Multi-threaded video processing
- Batch processing for multiple videos
- Config file for thresholds
- More unit tests

### Medium-term (1-4 weeks)
- ML-based exercise classifier (trained on video dataset)
- Form quality assessment (angle ranges, symmetry)
- Multiple person tracking
- Web UI for easier interaction

### Long-term (1-3 months)
- Real-time streaming support
- Mobile deployment (TFLite conversion)
- Cloud API deployment
- Exercise recommendation engine
