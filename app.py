import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Pickleball Pose AI", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to process video and save with annotations
def process_video(input_video_path):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output_video.mp4")

    cap = cv2.VideoCapture(input_video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()
    pose.close()

    return output_path


st.title("🏓 Pickleball Pose AI — CV Assignment")
st.write("Upload a video to analyze player pose using MediaPipe.")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Analyze video"):
        with st.spinner("Processing video... please wait ⏳"):
            # Save uploaded file to temp path
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, "input.mp4")

            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            output_path = process_video(input_path)

        st.success("✅ Video processed successfully!")
        st.video(output_path)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Download processed video",
                data=f,
                file_name="annotated_output.mp4",
                mime="video/mp4"
            )
