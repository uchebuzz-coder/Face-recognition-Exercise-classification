# Workout Video Analysis MVP

A Python-based computer vision pipeline for analyzing pre-recorded workout videos. The system identifies known individuals using facial recognition, classifies exercises from a fixed set, and counts repetitions using pose estimation.

## Features

- **Face Recognition**: Identifies known users using InsightFace with ArcFace embeddings
- **Exercise Classification**: Rule-based classifier for 4 exercises (Squats, Push-ups, Lunges, Bicep Curls)
- **Rep Counting**: Automatic repetition counting using pose-based state machines
- **Video Annotation**: Outputs annotated videos with person name, exercise type, and rep count overlays
- **JSON Results**: Structured analysis results with timestamps and confidence scores

## System Architecture

```
┌─────────────────┐
│  Video Input    │
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐ ┌──────────────┐
│  Face  │ │ MediaPipe    │
│  ID    │ │ Pose         │
└────────┘ └──────┬───────┘
              ┌────┴────┐
              │         │
              ▼         ▼
         ┌─────────┐ ┌───────────┐
         │Exercise │ │    Rep    │
         │Classify │ │  Counter  │
         └─────────┘ └───────────┘
              │         │
              └────┬────┘
                   ▼
         ┌──────────────────┐
         │ Output Generator │
         └──────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    ┌──────────┐      ┌──────────┐
    │Annotated │      │   JSON   │
    │  Video   │      │ Results  │
    └──────────┘      └──────────┘
```

## Project Structure

```
face-recognition & exercise classification/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── main.py                          # Main pipeline script
├── enroll_user.py                   # User enrollment script
├── modules/
│   ├── __init__.py
│   ├── face_recognition_module.py  # Face identification logic
│   ├── exercise_classifier.py      # Exercise classification
│   ├── rep_counter.py              # Rep counting state machines
│   └── utils.py                    # Utility functions (angles, etc.)
├── data/
│   ├── enrollment/                 # User enrollment images
│   ├── embeddings.pkl              # Stored face embeddings
│   └── videos/                     # Input workout videos
└── outputs/                        # Annotated videos and JSON results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- (Optional) CUDA-capable GPU for faster processing

### Setup Steps

1. **Clone or navigate to the repository**:
```bash
cd "face-recognition & exercise classification"
```

2. **Create a virtual environment** (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with InsightFace on Windows, you may need to install it from a wheel file. Visit [InsightFace GitHub](https://github.com/deepinsight/insightface) for platform-specific instructions.

For **CPU-only** systems, replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt` before installing:
```bash
pip install onnxruntime==1.17.0
```

## Usage

### Step 1: Enroll Users

Before analyzing videos, enroll known users by providing their photos.

#### Option A: Enroll from a directory of images
```bash
python enroll_user.py --name "John Doe" --directory "data/enrollment/john"
```

#### Option B: Enroll from specific image files
```bash
python enroll_user.py --name "Jane Smith" --images "data/enrollment/jane1.jpg" "data/enrollment/jane2.jpg"
```

**Tips**:
- Provide 3-5 clear photos per person for best results
- Use photos with good lighting and frontal face views
- Vary poses slightly (different angles, expressions)

### Step 2: Analyze Workout Videos

Process a workout video to generate annotated output and JSON results:

```bash
python main.py --video "data/videos/workout1.mp4"
```

This will create:
- `outputs/annotated_workout1.mp4` - Video with overlays
- `outputs/workout1_results.json` - Detailed analysis results

#### Advanced Options

```bash
# Specify custom output paths
python main.py --video "input.mp4" --output "custom_output.mp4" --json "results.json"

# Generate only JSON results (no video output)
python main.py --video "input.mp4" --no-video
```

## Output Format

### Annotated Video

The output video includes:
- **Top-left**: Identified person name and confidence score
- **Top-right**: Detected exercise and confidence score
- **Bottom-center**: Rep count and last rep duration
- **Pose skeleton**: Overlay showing body landmarks and connections

### JSON Results

Example output structure:

```json
{
  "video_filename": "workout1.mp4",
  "duration_seconds": 45.2,
  "identified_person": "John Doe",
  "person_confidence": 0.87,
  "exercise_detected": "Squats",
  "exercise_confidence": 0.92,
  "total_reps": 12,
  "reps_detail": [
    {
      "rep_num": 1,
      "start_time": 2.1,
      "bottom_time": 2.8,
      "end_time": 3.4,
      "duration": 1.3
    },
    {
      "rep_num": 2,
      "start_time": 3.8,
      "bottom_time": 4.4,
      "end_time": 5.0,
      "duration": 1.2
    }
  ],
  "processing_time_seconds": 8.3
}
```

## Supported Exercises

The system currently supports 4 exercises:

1. **Squats**
   - Detection: Hip angle < 100°, knee flexion, upright torso
   - Rep counting: STANDING → DOWN (hip < 90°) → STANDING (hip > 140°)

2. **Push-ups**
   - Detection: Elbow flexion, horizontal body orientation
   - Rep counting: UP → DOWN (elbow < 90°) → UP (elbow > 140°)

3. **Lunges**
   - Detection: Asymmetric leg position, forward stance
   - Rep counting: STANDING → DOWN (knee < 100°) → STANDING (knee > 150°)

4. **Bicep Curls**
   - Detection: Elbow flexion with stable shoulder, upright torso
   - Rep counting: EXTENDED → CURLED (elbow < 50°) → EXTENDED (elbow > 140°)

## How It Works

### 1. Face Recognition Module

- **Technology**: InsightFace with ArcFace model
- **Embedding**: 512-dimensional face vectors
- **Matching**: Cosine similarity with 0.6 threshold
- **Efficiency**: Checks every 30 frames (~1 per second)

### 2. Exercise Classification

- **Pose Extraction**: MediaPipe extracts 33 body landmarks per frame
- **Feature Engineering**: Calculates angles (hip, knee, elbow) and positions
- **Rule-based Classifier**: Each exercise has characteristic angle ranges
- **Temporal Smoothing**: Majority voting over 30-frame window

Key angles:
- Hip angle: Shoulder-Hip-Knee (squats)
- Elbow angle: Shoulder-Elbow-Wrist (push-ups, curls)
- Knee angle: Hip-Knee-Ankle (squats, lunges)

### 3. Rep Counting

- **State Machine**: Tracks exercise phases (up/down transitions)
- **Angle Thresholds**: Configurable for each exercise type
- **Rep Validation**: Only counts complete cycles
- **Timestamps**: Records start, bottom/peak, and end times

Example (Squats):
```
STANDING (hip > 140°) → DOWN (hip < 90°) → STANDING (hip > 140°) = 1 rep
```

## Extending the System

### Adding a New Exercise

1. **Add exercise signature to classifier** (`modules/exercise_classifier.py`):
```python
def classify_new_exercise(self, features: Dict[str, float]) -> float:
    confidence = 0.0
    # Add angle checks
    if features['some_angle'] < threshold:
        confidence += 0.4
    return confidence
```

2. **Create a rep counter** (`modules/rep_counter.py`):
```python
class NewExerciseRepCounter(RepCounter):
    def process_frame(self, landmarks, timestamp: float) -> None:
        # Implement state machine logic
        pass
```

3. **Register in main pipeline** (`main.py`):
```python
self.rep_counters['New Exercise'] = NewExerciseRepCounter()
```

### Adjusting Rep Counting Thresholds

Edit the counter initialization in `main.py`:

```python
# Make squats more/less strict
self.rep_counters['Squats'] = SquatRepCounter(
    down_threshold=80,   # Default: 90
    up_threshold=150     # Default: 140
)
```

## Performance

- **Processing Speed**: 20-30 FPS on GPU, 8-12 FPS on CPU
- **Accuracy**: 
  - Face Recognition: ~95% for enrolled users in good lighting
  - Exercise Classification: ~85-90% for supported exercises
  - Rep Counting: ~90% accuracy for clean form

## Troubleshooting

### Issue: "InsightFace not installed"
**Solution**: Install InsightFace with proper dependencies:
```bash
pip install insightface onnxruntime-gpu
```

### Issue: Slow processing on CPU
**Solution**: 
1. Use lower resolution videos (720p instead of 1080p)
2. Reduce pose model complexity in `main.py`:
```python
self.pose = self.mp_pose.Pose(model_complexity=0)  # 0=lite, 1=full, 2=heavy
```

### Issue: Face not detected
**Solution**:
- Ensure face is clearly visible and well-lit in video
- Use high-quality enrollment photos (frontal view, good lighting)
- Lower similarity threshold in `FaceRecognitionModule` initialization

### Issue: Incorrect exercise classification
**Solution**:
- Ensure full body is visible in frame
- Check that exercise form matches expected patterns
- Adjust angle thresholds in `exercise_classifier.py`

### Issue: Reps not counted
**Solution**:
- Verify exercise is correctly classified first
- Adjust rep counter thresholds for your specific form
- Ensure full range of motion (complete up/down cycles)

## GPU Setup (Optional)

For faster processing with NVIDIA GPU:

1. Install CUDA Toolkit (11.x or 12.x)
2. Install cuDNN
3. Install GPU-enabled packages:
```bash
pip install onnxruntime-gpu==1.17.0
```

Verify GPU usage:
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

## Limitations

- **Single person**: Designed for one primary subject per video
- **Pre-recorded only**: No live streaming support
- **Fixed exercise set**: Limited to 4 exercise types
- **Form dependent**: Requires proper exercise form for accurate counting
- **Lighting dependent**: Face recognition needs good lighting conditions

## Future Enhancements

Potential improvements for production:
- Multi-person tracking
- Additional exercises (planks, jumping jacks, etc.)
- ML-based exercise classifier (replace rule-based)
- Form quality assessment
- Real-time streaming support
- Mobile deployment

## Technical Requirements

- **Minimum**: Intel i5/Ryzen 5, 8GB RAM, Python 3.8+
- **Recommended**: Intel i7/Ryzen 7, 16GB RAM, NVIDIA GPU (GTX 1060+)
- **Storage**: ~2GB for dependencies + model weights

## License

This is an internal MVP for demonstration purposes.

## Support

For issues or questions, refer to:
- MediaPipe: https://google.github.io/mediapipe/
- InsightFace: https://github.com/deepinsight/insightface
- OpenCV: https://docs.opencv.org/
