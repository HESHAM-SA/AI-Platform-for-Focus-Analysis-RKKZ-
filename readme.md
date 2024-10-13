# Student Focus Tracker

This project uses computer vision and machine learning to monitor student focus during video lectures. It analyzes facial landmarks and eye movements to compute a focus score based on the student's gaze direction and blink rate.

## Features

- **Face and Eye Tracking**: Utilizes MediaPipe for real-time face and eye landmark detection.
- **Focus Scoring**: Calculates a focus score based on gaze direction and blink detection.
- **Real-Time Analysis**: Processes webcam input or uploaded videos to assess concentration levels.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- Streamlit
- Streamlit-WebRTC
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
