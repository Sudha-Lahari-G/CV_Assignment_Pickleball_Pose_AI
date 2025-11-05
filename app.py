import tempfile
import cv2
import numpy as np
import streamlit as st

# Lazy import mediapipe to speed initial load
def lazy_import_mediapipe():
    import importlib
    return importlib.import_module("mediapipe")


from utils.geometry import angle_3pts, rad2deg, safe_min, safe_mean

st.set_page_config(page_title="Pickleball AI Feedback (Pose Demo)", page_icon="🏓", layout="wide")
st.title("🏓 Pickleball AI Feedback — Pose-based Demo")
st.caption("Upload a short stroke/serve video (<= 3 min). The app will overlay pose landmarks and generate 1–2 basic coaching tips.")

with st.expander("How it works"):
    st.markdown("""
    - Uses **MediaPipe Pose** to extract a 33-landmark body skeleton per frame.
    - Computes a few simple angles (knee, elbow, trunk) and applies lightweight rules to generate feedback.
    - Overlays the detected pose on the uploaded video and produces an annotated MP4.
    """)

uploaded = st.file_uploader("Upload a video file (MP4/MOV/MKV recommended)", type=["mp4","mov","mkv","avi"])
process_btn = st.button("Analyze video", disabled=uploaded is None)

def draw_landmarks(image, pose_landmarks, mp_drawing, mp_pose):
    mp_drawing.draw_landmarks(
        image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2))
    return image

def get_landmark_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def analyze_metrics(metrics):
    """Return 1–2 concise feedback strings based on measured angles/positions."""
    feedback = []

    # Rules
    # 1) Knee bend (use smaller of L/R knee min angle across motion)
    min_knee = safe_min([metrics.get("min_knee_L"), metrics.get("min_knee_R")])
    if min_knee is not None:
        if min_knee > 165:
            feedback.append("Try bending your knees more to load power (target ~140°–160° at the lowest point).")
        elif min_knee < 120:
            feedback.append("Avoid over-squatting; keep knee bend moderate for balance (~135°–165°).")

    # 2) Elbow extension at contact (smaller of L/R)
    min_elbow = safe_min([metrics.get("elbow_at_contact_L"), metrics.get("elbow_at_contact_R")])
    if min_elbow is not None and min_elbow < 140:
        feedback.append("Extend your hitting arm a bit more at contact for a cleaner strike.")

    # 3) Trunk tilt (hip-shoulder line relative to vertical) average
    avg_trunk_tilt = metrics.get("avg_trunk_tilt")
    if avg_trunk_tilt is not None and avg_trunk_tilt > 25:
        feedback.append("Keep your upper body a little more upright to improve control (reduce trunk tilt).")

    # Keep 1–2 comments
    if len(feedback) == 0:
        feedback.append("Nice form! Keep a steady base and smooth swing through contact.")
    return feedback[:2]

if process_btn and uploaded is not None:
    mp = lazy_import_mediapipe()
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Save upload to temp file
    tdir = tempfile.mkdtemp()
    in_path = os.path.join(tempfile.gettempdir(), "input_video")
    suffix = pathlib.Path(uploaded.name).suffix.lower()
    if suffix not in [".mp4",".mov",".mkv",".avi"]:
        suffix = ".mp4"
    in_path += suffix
    with open(in_path, "wb") as f:
        f.write(uploaded.read())

    # OpenCV reader
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("Could not open the video. Try another format or re-encode to MP4 (H.264).")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    # Prepare writer
    out_path = os.path.join(tdir, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Metrics storage
    knee_angles_L = []
    knee_angles_R = []
    elbow_angles_L = []
    elbow_angles_R = []
    trunk_tilts = []

    # Pick frames to estimate "contact" as frames with peak wrist speed proxy; here, we'll just use mid-frame as proxy
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    contact_frame_idx = max(0, total_frames // 2)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR->RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                lms = results.pose_landmarks.landmark

                # Indices from MediaPipe Pose
                # Hips/Knees/Ankles
                L_HIP, R_HIP = mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value
                L_KNEE, R_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value
                L_ANK, R_ANK = mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value
                # Shoulders/Elbows/Wrists
                L_SH, R_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                L_EL, R_EL = mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value
                L_WR, R_WR = mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value

                # Points
                lhip = get_landmark_xy(lms, L_HIP, w, h)
                rhip = get_landmark_xy(lms, R_HIP, w, h)
                lknee = get_landmark_xy(lms, L_KNEE, w, h)
                rknee = get_landmark_xy(lms, R_KNEE, w, h)
                lank = get_landmark_xy(lms, L_ANK, w, h)
                rank = get_landmark_xy(lms, R_ANK, w, h)
                lsh = get_landmark_xy(lms, L_SH, w, h)
                rsh = get_landmark_xy(lms, R_SH, w, h)
                lel = get_landmark_xy(lms, L_EL, w, h)
                rel = get_landmark_xy(lms, R_EL, w, h)
                lwr = get_landmark_xy(lms, L_WR, w, h)
                rwr = get_landmark_xy(lms, R_WR, w, h)

                # Angles (in degrees)
                knee_L = rad2deg(angle_3pts(lhip, lknee, lank))
                knee_R = rad2deg(angle_3pts(rhip, rknee, rank))
                elbow_L = rad2deg(angle_3pts(lsh, lel, lwr))
                elbow_R = rad2deg(angle_3pts(rsh, rel, rwr))

                knee_angles_L.append(knee_L)
                knee_angles_R.append(knee_R)
                elbow_angles_L.append(elbow_L)
                elbow_angles_R.append(elbow_R)

                # Trunk tilt: angle between vertical and shoulder-hip line (use left side)
                # Compute vector hip->shoulder
                vec = lsh - lhip
                # Angle from vertical: arctan of dx/dy
                if np.linalg.norm(vec) > 1e-6:
                    tilt_rad = np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6)
                    trunk_tilts.append(rad2deg(tilt_rad))

                # Draw landmarks
                frame = draw_landmarks(frame, results.pose_landmarks, mp_drawing, mp_pose)

            writer.write(frame)
            idx += 1

    writer.release()
    cap.release()

    # Aggregate metrics
    metrics = {
        "min_knee_L": float(np.min(knee_angles_L)) if knee_angles_L else None,
        "min_knee_R": float(np.min(knee_angles_R)) if knee_angles_R else None,
        "elbow_at_contact_L": float(elbow_angles_L[min(len(elbow_angles_L)-1, contact_frame_idx)]) if elbow_angles_L else None,
        "elbow_at_contact_R": float(elbow_angles_R[min(len(elbow_angles_R)-1, contact_frame_idx)]) if elbow_angles_R else None,
        "avg_trunk_tilt": float(np.mean(trunk_tilts)) if trunk_tilts else None,
    }

    st.subheader("Results")
    st.write("**Key metrics (approximate):**", metrics)
    fb = analyze_metrics(metrics)
    st.success("**Feedback**")
    for f in fb:
        st.markdown(f"- {f}")

    st.subheader("Annotated video")
    with open(out_path, "rb") as vf:
        st.video(vf.read())
else:
    st.info("Upload a video to enable the Analyze button.")
