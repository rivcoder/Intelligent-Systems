## Motion Analyzer

Overview

Motion Analyzer is a physics-based web application that analyzes motion from video input. It uses computer vision techniques to track an object and compute position, velocity, and acceleration over time.

Features

- Video upload and preview
- Object tracking using OpenCV
- Pixel-to-meter calibration
- Time calculation using FPS
- Noise reduction using smoothing filters
- Graph visualization using Plotly
- Physics-based interpretation
- Lab report generation (PDF)

Technologies Used

- Frontend: HTML, CSS, JavaScript
- Backend: Python (Flask)
- Computer Vision: OpenCV
- Graphing: Plotly.js
- Report Generation: jsPDF

Working Principle

1. The user uploads a video of motion.
2. The system detects and tracks the object frame-by-frame using computer vision.
3. Pixel coordinates are converted into real-world measurements using calibration.
4. Position data is processed and smoothed to reduce noise.
5. Velocity and acceleration are calculated using numerical differentiation.
6. Graphs are generated to visualize motion.
7. The system provides a physics-based explanation and analysis.

Applications

- Physics experiments (free fall, projectile motion)
- Educational visualization
- Motion analysis in laboratory setups

Limitations

- Works best with high contrast objects (e.g., red ball)
- Requires stable camera
- Accuracy depends on calibration and video quality
