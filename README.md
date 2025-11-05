# Pickleball AI Feedback (Pose Demo)

A minimal demo that lets a user upload a 2–3 min pickleball stroke/serve video, runs **MediaPipe Pose** on it, overlays the skeleton, computes a few simple angles, and returns 1–2 feedback comments based on basic rules.

## ✨ What this covers
- Video upload UI (**Streamlit**)
- Pose detection (**MediaPipe Pose**)
- Simple metrics (knee angle, elbow extension, trunk tilt)
- Lightweight feedback rules
- Display original + annotated video and feedback on the same page

## 🧰 Tech
- MediaPipe, OpenCV, Streamlit, NumPy
- (Optional) `pyngrok` to expose the Streamlit UI from Google Colab

---

## 🚀 Run on Google Colab (recommended)

> **Tip:** Colab works best with short MP4s (H.264). If your source is from YouTube, trim it first locally or in Colab.

1. **Open Colab** and create a new notebook.
2. **Install dependencies**:
   ```python
   !pip -q install streamlit==1.39.0 opencv-python==4.10.0.84 mediapipe==0.10.14 pyngrok==7.2.3
   ```
3. **Upload the project files** (drag-and-drop the zip you downloaded or clone your repo):
   ```python
   from google.colab import files
   uploaded = files.upload()  # upload pickleball_ai_feedback.zip
   !unzip -o pickleball_ai_feedback.zip -d /content/
   %cd /content/pickleball_ai_feedback
   ```
4. **Start Streamlit and tunnel with ngrok**:
   ```python
   # 1) launch streamlit (non-blocking)
   import threading, subprocess, time, os
   def run_streamlit():
       subprocess.call(["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"])
   t = threading.Thread(target=run_streamlit, daemon=True); t.start()
   time.sleep(3)

   # 2) expose with ngrok
   from pyngrok import ngrok
   public_url = ngrok.connect(8501, "http")
   public_url
   ```
   Click the URL printed by the last line to open your Streamlit app.
5. **Use the app**: upload your stroke/serve video (<= 3 min), wait for processing, then see the **feedback** and **annotated video**.

> If video fails to render, re-encode to MP4 (H.264) using e.g. HandBrake or `ffmpeg -i input.mov -vcodec libx264 -acodec aac out.mp4`.

---

## 🧪 Sample video
The assignment references this video: `https://www.youtube.com/watch?v=jnlpyUHRq4I`. 
Download and trim a serve or stroke segment (2–3 min) before uploading in the app.

---

## 🧠 Feedback rules (simple, tweakable)
- **Knee bend**: If minimum knee angle > 165°, suggest “bend your knees more.”
- **Elbow at contact**: If elbow angle at mid-swing < 140°, suggest “extend your arm more at contact.”
- **Trunk tilt**: If avg tilt > 25°, suggest “stay a bit more upright.”

You can adjust thresholds in `app.py` to change strictness.

---

## 📦 Repo structure
```
pickleball_ai_feedback/
├── app.py
├── requirements.txt
└── utils/
    └── geometry.py
```

---

## 📝 Submission checklist
- [ ] Push code to **GitHub** (this folder).  
- [ ] Add a **README** (this file) with setup/run steps.  
- [ ] Include a **screenshot** of the running app with your own video.  
- [ ] (Optional) Record a short **demo clip** of the annotated output.

---

## 🔧 Local dev (optional)
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501.
