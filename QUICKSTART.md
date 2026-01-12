# Quick Start Guide

Get up and running with workout video analysis in 5 minutes.

## 1. Install Dependencies (2 minutes)

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

## 2. Enroll Users (1 minute)

Create a folder with 3-5 photos of yourself:

```bash
# Create user directory
mkdir "data/enrollment/YourName"

# Copy your photos there (or use file explorer)
# Then enroll:
python enroll_user.py --name "YourName" --directory "data/enrollment/YourName"
```

## 3. Process Video (2 minutes)

Place a workout video in `data/videos/` and run:

```bash
python main.py --video "data/videos/your_workout.mp4"
```

Check results in `outputs/` folder:
- Annotated video with overlays
- JSON file with detailed results

## Done!

Your video is analyzed with:
- ✓ Person identification
- ✓ Exercise classification
- ✓ Rep counting
- ✓ Timestamped results

## Next Steps

- Read full [README.md](README.md) for details
- Adjust thresholds for better accuracy
- Add more users
- Try different exercises

## Minimal Example

```bash
# One-liner after setup
python main.py --video "data/videos/squats.mp4"
```

That's it! 🎉
