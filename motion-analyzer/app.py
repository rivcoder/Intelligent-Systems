from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import uuid

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
# OBJECT DETECTION (RED BALL)
# =========================

KERNEL_OPEN = np.ones((3, 3), np.uint8)
KERNEL_DILATE = np.ones((5, 5), np.uint8)
LOWER_RED1 = np.array([0, 120, 70], dtype=np.uint8)
UPPER_RED1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED2 = np.array([170, 120, 70], dtype=np.uint8)
UPPER_RED2 = np.array([180, 255, 255], dtype=np.uint8)


def _safe_savgol(arr, max_window=11, poly=2):
    n = len(arr)
    if n < 5:
        return np.asarray(arr)

    window = min(max_window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return np.asarray(arr)

    poly = min(poly, window - 1)
    return savgol_filter(arr, window, poly)


def _safe_savgol_derivative(arr, dt, deriv_order, max_window=15, poly=3):
    n = len(arr)
    if n < 7 or dt <= 0:
        if deriv_order == 1:
            return np.gradient(arr, dt if dt > 0 else 1.0)
        return np.gradient(np.gradient(arr, dt if dt > 0 else 1.0), dt if dt > 0 else 1.0)

    window = min(max_window, n if n % 2 == 1 else n - 1)
    if window < 7:
        if deriv_order == 1:
            return np.gradient(arr, dt)
        return np.gradient(np.gradient(arr, dt), dt)

    poly = min(poly, window - 1)
    if poly <= deriv_order:
        poly = deriv_order + 1
        if poly >= window:
            if deriv_order == 1:
                return np.gradient(arr, dt)
            return np.gradient(np.gradient(arr, dt), dt)

    return savgol_filter(arr, window_length=window, polyorder=poly, deriv=deriv_order, delta=dt)


def _moving_average(arr, window=5):
    if window <= 1 or len(arr) < window:
        return np.asarray(arr)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(arr, kernel, mode="same")


def _longest_track_segment(valid_idx, max_gap=2):
    if len(valid_idx) == 0:
        return None
    best_start = 0
    best_end = 0
    run_start = 0
    for i in range(1, len(valid_idx)):
        if valid_idx[i] - valid_idx[i - 1] > (max_gap + 1):
            if (i - 1 - run_start) > (best_end - best_start):
                best_start, best_end = run_start, i - 1
            run_start = i
    if (len(valid_idx) - 1 - run_start) > (best_end - best_start):
        best_start, best_end = run_start, len(valid_idx) - 1
    return int(valid_idx[best_start]), int(valid_idx[best_end])


def detect_ball(frame, roi=None, min_area=120):
    if roi is not None:
        x0, y0, x1, y1 = roi
        view = frame[y0:y1, x0:x1]
        if view.size == 0:
            return None
    else:
        x0, y0 = 0, 0
        view = frame

    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)

    # 🎯 RED color detection (more reliable)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)

    mask = mask1 + mask2

    # noise removal
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask = cv2.dilate(mask, KERNEL_DILATE, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > min_area:
            m = cv2.moments(c)
            if m["m00"] <= 0:
                return None
            x = int(m["m10"] / m["m00"])
            y = int(m["m01"] / m["m00"])
            (_, _), r = cv2.minEnclosingCircle(c)

            if r > 5:
                return int(x + x0), int(y + y0)

    return None


def track_with_optical_flow(prev_gray, gray, prev_point):
    if prev_gray is None or gray is None or prev_point is None:
        return None

    p0 = np.array([[prev_point]], dtype=np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    if p1 is None or st is None or st[0][0] != 1:
        return None

    if err is not None and float(err[0][0]) > 35:
        return None

    x, y = p1[0][0]
    return int(round(x)), int(round(y))


def detect_motion_blob(prev_gray, gray, min_area=150):
    if prev_gray is None or gray is None:
        return None
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    m = cv2.moments(c)
    if m["m00"] <= 0:
        return None
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return x, y

# =========================
# MOTION ANALYSIS
# =========================

def analyze_motion(video_path, ref_pixels, ref_meters):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 10 or fps > 120:
        fps = 30

    raw_positions = []
    all_times = []
    per_frame_pos = []

    frame_no = 0
    prev_x, prev_y = None, None
    prev_gray = None
    lost_count = 0
    max_jump = 80

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = (t_msec / 1000.0) if t_msec and t_msec > 0 else (frame_no / fps)
        if all_times and t <= all_times[-1]:
            t = all_times[-1] + (1.0 / fps)
        all_times.append(t)

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow_obj = None
        if prev_x is not None and prev_y is not None and prev_gray is not None and lost_count < 15:
            flow_obj = track_with_optical_flow(prev_gray, gray, (prev_x, prev_y))
            if flow_obj is not None:
                fx, fy = flow_obj
                if fx < 0 or fy < 0 or fx >= w or fy >= h:
                    flow_obj = None

        roi = None
        if prev_x is not None and lost_count < 10:
            pad = 120
            roi = (
                max(0, prev_x - pad),
                max(0, prev_y - pad),
                min(w, prev_x + pad),
                min(h, prev_y + pad)
            )

        # Prefer color detector; fall back to motion blob or optical flow.
        det_obj = detect_ball(frame, roi=roi, min_area=80)
        if det_obj is None:
            det_obj = detect_ball(frame, roi=None, min_area=120)

        motion_obj = detect_motion_blob(prev_gray, gray, min_area=120)

        if det_obj is not None:
            obj = det_obj
        elif motion_obj is not None:
            obj = motion_obj
        else:
            obj = flow_obj

        if obj:
            x, y = obj

            # 🔥 TRACKING STABILITY
            if prev_x is not None:
                dist = np.hypot((x - prev_x), (y - prev_y))

                # ignore sudden jumps (noise)
                if dist > max_jump:
                    obj = None
                else:
                    prev_x, prev_y = x, y
            else:
                prev_x, prev_y = x, y

        if obj:
            raw_positions.append(h - y)
            per_frame_pos.append(float(h - y))
            lost_count = 0
        else:
            per_frame_pos.append(np.nan)
            lost_count += 1

        prev_gray = gray
        frame_no += 1

    cap.release()

    quality_warnings = []

    # 🚨 detection failure (very low data)
    if len(raw_positions) < 10:
        return None, "Object tracking failed. Almost no frames detected the object."

    per_frame_pos = np.array(per_frame_pos, dtype=np.float64)
    all_times = np.array(all_times, dtype=np.float64)
    if len(all_times) < 2:
        return None, "Video timing information is invalid."

    valid_idx = np.where(~np.isnan(per_frame_pos))[0]
    segment = _longest_track_segment(valid_idx, max_gap=2)
    if segment is None:
        return None, "Object tracking failed. No stable motion segment found."
    start_i, end_i = segment

    seg_pos = per_frame_pos[start_i:end_i + 1]
    seg_times = all_times[start_i:end_i + 1]
    seg_valid = np.where(~np.isnan(seg_pos))[0]

    if len(seg_valid) < 25:
        quality_warnings.append(
            "Short tracking segment (<25 frames). Graphs may be noisy; focus on overall trends, not exact values."
        )

    missing_ratio = float(np.mean(np.isnan(seg_pos)))
    if missing_ratio > 0.4:
        quality_warnings.append(
            f"Many missed detections (~{missing_ratio*100:.0f}% frames). Curves may be less accurate; improve lighting/background."
        )

    # Interpolate only small internal gaps in the selected stable segment.
    positions = np.interp(np.arange(len(seg_pos)), seg_valid, seg_pos[seg_valid])
    times = seg_times

    # =========================
    # SMOOTHING (STRONG UPGRADE)
    # =========================

    try:
        positions = _safe_savgol(positions, max_window=15, poly=3)
    except Exception:
        # fallback if data small
        positions = np.convolve(positions, np.ones(5) / 5, mode='same')

    # =========================
    # SCALING
    # =========================

    if ref_pixels <= 0 or ref_meters <= 0:
        return None, "Invalid calibration values."

    scale = ref_meters / ref_pixels
    pos = positions * scale

    # =========================
    # PHYSICS
    # =========================
    if len(times) < 3:
        quality_warnings.append("Not enough samples for smooth derivatives; curves are very approximate.")
    if float(np.median(np.diff(times))) <= 0:
        quality_warnings.append("Frame timing is irregular; derivatives may be distorted.")

    # Derivatives on real timestamps are more robust when frame spacing is imperfect.
    vel = np.gradient(pos, times)
    vel = _safe_savgol(vel, max_window=11, poly=2)
    vel = _moving_average(vel, window=5)

    acc = np.gradient(vel, times)
    acc = _safe_savgol(acc, max_window=11, poly=2)
    acc = _moving_average(acc, window=5)

    # Keep clipping wide so we only remove non-physical spikes.
    vel = np.clip(vel, -100, 100)
    acc = np.clip(acc, -200, 200)

    # =========================
    # SCALE DIAGNOSTICS
    # =========================
    px_span = float(np.nanmax(positions) - np.nanmin(positions))
    m_span = float(np.nanmax(pos) - np.nanmin(pos))
    g_est = float(np.nanmedian(np.abs(acc)))
    v_peak = float(np.nanmax(np.abs(vel)))
    a_peak = float(np.nanmax(np.abs(acc)))
    scale_notes = []
    if px_span < (0.15 * ref_pixels):
        scale_notes.append(
            "Tracked motion span is very small compared to reference length, so calibration noise can dominate."
        )
    if g_est > 30:
        scale_notes.append(
            "Estimated acceleration is unusually high. Re-check reference pixels and camera angle."
        )
    if g_est < 0.3:
        scale_notes.append(
            "Estimated acceleration is unusually low. Re-check reference pixels and ensure clear motion in video."
        )
    
    # =========================
    # GRAPH
    # =========================

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times.tolist(),
        y=pos.tolist(),
        name="Position (m)"
    ))

    # Optional visual reference bar to make scaling obvious.
    if ref_meters > 0 and times.size >= 2:
        fig.add_trace(go.Scatter(
            x=[times[0], times[-1]],
            y=[ref_meters, ref_meters],
            name="Reference distance",
            mode="lines",
            line=dict(dash="dash", color="rgba(255,255,255,0.4)")
        ))

    fig.add_trace(go.Scatter(
        x=times.tolist(),
        y=vel.tolist(),
        name="Velocity (m/s)"
    ))

    fig.add_trace(go.Scatter(
        x=times.tolist(),
        y=acc.tolist(),
        name="Acceleration (m/s²)"
    ))

    fig.update_layout(
        title="Motion Analysis",
        xaxis_title="Time (s)",
        yaxis_title="Value"
    )

    explanation = (
        "Motion analysis completed.\n\n"
        "Calibration:\n"
        f"- Scale factor: {scale:.6f} m/pixel\n"
        f"- Tracked vertical span: {px_span:.2f} px ({m_span:.3f} m)\n\n"
        "Estimated Results:\n"
        f"- Peak |velocity|: {v_peak:.3f} m/s\n"
        f"- Median |acceleration|: {g_est:.3f} m/s² (approximate)\n"
        f"- Peak |acceleration|: {a_peak:.3f} m/s²\n\n"
        "Interpretation:\n"
        "- Position-Time: displacement trend\n"
        "- Velocity-Time: slope of position\n"
        "- Acceleration-Time: slope of velocity\n\n"
        "Notes:\n"
        "- Acceleration may deviate from expected gravitational value (~9.8 m/s²) due to calibration, tracking noise, or smoothing.\n"
        "- Results are approximate and depend on video quality and setup.\n\n"
        "Tips:\n"
        "- Keep camera fixed and perpendicular to motion plane\n"
        "- Use reference length in the same plane\n"
        "- Ensure good lighting and high contrast object"
    )
    if scale_notes:
        explanation += "\n\nScale checks:\n- " + "\n- ".join(scale_notes)

    if quality_warnings:
        explanation += "\n\nTracking warnings:\n- " + "\n- ".join(quality_warnings)

    return fig.to_dict(), explanation

# =========================
# API
# =========================

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"})

        file = request.files["video"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"})

        ref_pixels = float(request.form["ref_pixels"])
        ref_meters = float(request.form["ref_meters"])

        # safe unique filename
        filename = str(uuid.uuid4()) + ".mp4"
        path = os.path.join(UPLOAD_FOLDER, filename)

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
