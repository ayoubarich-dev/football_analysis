# ⚽ Football Analysis System

An AI-powered football match analysis system that detects and tracks players, referees, and the ball, then calculates real-time statistics using computer vision and deep learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📺 Demo

🔗 **GitHub Repository**: [ayoubarich-dev/football_analysis](https://github.com/ayoubarich-dev/football_analysis)

## 🎯 Features

- **Player & Ball Detection**: YOLOv5-based object detection
- **Multi-Object Tracking**: Track players, referees, and ball across frames
- **Team Assignment**: Automatic team identification using jersey color clustering
- **Ball Possession**: Calculate which player has the ball and team possession percentages
- **Camera Movement Compensation**: Adjust for camera pan and zoom
- **Speed & Distance Tracking**: Measure player speed (km/h) and distance covered (m)
- **Top-Down View Transformation**: Convert pixel coordinates to real-world field positions

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ayoubarich-dev/football_analysis.git
cd football_analysis

# Install dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics opencv-python supervision scikit-learn pandas numpy tqdm
```

## 🚀 Quick Start

1. **Place your video** in the `input_videos/` folder

2. **Run the analysis**:
```bash
python main.py
```

3. **Output** will be saved to `output_videos/output_video.avi`

## 📁 Project Structure

```
├── trackers/                    # Object detection and tracking
├── team_assigner/              # Team identification by jersey color
├── player_ball_assigner/       # Ball possession detection
├── camera_movement_estimator/  # Camera motion compensation
├── view_transformer/           # Perspective transformation
├── speed_and_distance_estimator/ # Speed and distance calculation
├── utils/                      # Utility functions
├── models/                     # Trained YOLO model (best.pt)
├── input_videos/               # Input video files
├── output_videos/              # Processed output videos
└── stubs/                      # Cached tracking data
```

## 🎓 Training Your Own Model

Use the provided Jupyter notebook:

```bash
jupyter notebook training/football_training_yolo_v5.ipynb
```

The notebook includes:
- Dataset download from Roboflow
- YOLOv5 training configuration
- Model evaluation

## 🔧 How It Works

1. **Detection**: YOLOv5 detects players, referees, and ball
2. **Tracking**: ByteTrack assigns unique IDs to each object
3. **Team Assignment**: K-means clustering on jersey colors
4. **Camera Adjustment**: Optical flow to compensate camera movement
5. **Position Transformation**: Perspective transform to field coordinates
6. **Statistics**: Calculate speed, distance, and possession

## 📊 Output Visualizations

- Player tracking with team colors
- Ball possession indicators
- Speed and distance stats per player
- Team possession percentages
- Camera movement overlay

## ⚙️ Configuration

Key parameters in the code:

- **Frame rate**: 24 fps (default)
- **Speed calculation window**: 5 frames
- **Ball assignment distance**: 70 pixels
- **Field dimensions**: 68m x 23.32m

## 🎥 Sample Output

The system annotates the video with:
- ⚪ Ellipses under players (colored by team)
- 🔺 Triangle markers for ball and ball possession
- 📊 Real-time statistics overlay
- 📹 Camera movement indicators

## 📝 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Trained YOLOv5 model (`models/best.pt`)

## 🤝 Credits

- **YOLOv5/Ultralytics**: Object detection
- **ByteTrack**: Multi-object tracking
- **Roboflow**: Training dataset
- **OpenCV**: Computer vision operations

## 👨‍💻 Author

**Ayoub Arich** - [@ayoubarich-dev](https://github.com/ayoubarich-dev)

## 📄 License

MIT License - feel free to use for your projects!

## 🐛 Troubleshooting

**Issue**: Model not found
- Ensure `models/best.pt` exists or train a new model

**Issue**: Slow processing
- Enable stub files: `read_from_stub=True`
- Reduce video resolution
- Use GPU acceleration

**Issue**: Poor team assignment
- Adjust lighting conditions
- Ensure clear jersey colors
- Check K-means cluster count

## 🚀 Future Improvements

- [ ] Real-time processing
- [ ] Heatmap generation
- [ ] Pass detection
- [ ] Offside detection
- [ ] Multi-camera support
