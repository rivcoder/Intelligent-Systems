from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import plotly.graph_objs as go

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# SERVE FILES
# =========================

@app.route("/")
def home():
    return send_from_directory('.', 'index.html')

@app.route("/style.css")
def style():
    return send_from_directory('.', 'style.css')

@app.route("/script.js")
def script():
    return send_from_directory('.', 'script.js')


# =========================
# DETECT RED OBJECT 
# =========================

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 🎯 detect ANY bright + saturated object (not color-specific)
    lower = np.array([0, 80, 80])     # ignore dull/gray areas
    upper = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # noise removal (same as before)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # pick largest visible object
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 150:   # slightly stricter (avoid noise)
            (x, y), r = cv2.minEnclosingCircle(c)

            if r > 5:  # ignore tiny junk
                return int(x), int(y)

    return None

# =========================
# MOTION ANALYSIS 
# =========================

def analyze_motion(video_path, ref_pixels, ref_meters):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    positions = []
    times = []

    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        obj = detect_ball(frame)

        if obj:
            x, y = obj
            h = frame.shape[0]

            positions.append(h - y)
            times.append(frame_no / fps)

        frame_no += 1

    cap.release()

    # 🚨 If detection failed
    if len(positions) < 15:
        return None, "Ball not detected properly. Use clear red ball."

    positions = np.array(positions)
    times = np.array(times)

    # 🔥 Smooth (VERY IMPORTANT)
    window = 7
    positions = np.convolve(positions, np.ones(window)/window, mode='valid')
    times = times[:len(positions)]

    # scale to meters
    scale = ref_meters / ref_pixels
    pos = positions * scale

    # physics
    vel = np.gradient(pos, times)
    acc = np.gradient(vel, times)

    # smooth velocity & acceleration
    vel = np.convolve(vel, np.ones(5)/5, mode='same')
    acc = np.convolve(acc, np.ones(5)/5, mode='same')

    # =========================
    # GRAPH
    # =========================

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=times.tolist(), y=pos.tolist(), name="Position (m)"))
    fig.add_trace(go.Scatter(x=times.tolist(), y=vel.tolist(), name="Velocity (m/s)"))
    fig.add_trace(go.Scatter(x=times.tolist(), y=acc.tolist(), name="Acceleration (m/s²)"))

    fig.update_layout(
        title="Motion Analysis",
        xaxis_title="Time (s)",
        yaxis_title="Value"
    )

    return fig.to_dict(), """
    The generated graphs represent the motion of the object with respect to time, 
    showing how position, velocity, and acceleration vary during the observed interval.
    The position–time graph illustrates the displacement of the object. 
    A smoother curve indicates continuous motion, while steeper regions correspond to higher velocities.
    The velocity–time graph represents the rate of change of position. 
    Variations in this graph indicate changes in speed or direction of motion.
    The acceleration–time graph shows how velocity changes over time. 
    Non-zero acceleration suggests the presence of forces acting on the object.
    From a physics perspective:
    - Velocity is the derivative of position with respect to time
    - Acceleration is the derivative of velocity with respect to time.
<br><br><br>
    It is important to note that the analysis is based on video 
    tracking and pixel-to-distance calibration. Due to factors such as 
    frame rate limitations, resolution, tracking noise, and scaling approximations, 
    the results may not be perfectly accurate.
    However, the graphs correctly represent the overall motion trends 
    and provide a reliable qualitative understanding of the object's kinematics.
    """

# =========================
# API
# =========================

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["video"]
        ref_pixels = float(request.form["ref_pixels"])
        ref_meters = float(request.form["ref_meters"])

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        graph, explanation = analyze_motion(path, ref_pixels, ref_meters)

        if graph is None:
            return jsonify({"error": explanation})

        return jsonify({
            "data": graph["data"],
            "layout": graph["layout"],
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)