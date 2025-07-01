# app.py

import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import yt_dlp
from skimage.metrics import structural_similarity as ssim

# Configure Streamlit page
st.set_page_config(page_title="Video Frame Extractor", layout="wide")
st.title("Video Frame Extractor (Faces + Motion Filtering)")

# -------------------------- Function Definitions --------------------------

def download_youtube_video(url: str) -> str:
    """
    Downloads a YouTube video using yt-dlp.
    - Saves the video in a temporary directory.
    - Returns the full path to the downloaded .mp4 file.
    """
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "video.%(ext)s")
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'noplaylist': True,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path.replace("%(ext)s", "mp4")


def save_uploaded_video(uploaded_file) -> str:
    """
    Saves a user-uploaded video file to a temporary directory.
    - Returns the full file path where the video is saved.
    """
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    return video_path


def load_video_metadata(video_path: str) -> tuple:
    """
    Loads basic information about the video.
    - Returns total frame count, frames per second (fps), and video duration in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()
    return total_frames, fps, duration


def extract_frames_with_faces_and_motion_filter(
    video_path: str,
    start_time: float,
    end_time: float,
    time_gap: float,
    motion_threshold: float = 0.80
) -> list:
    """
    Extracts frames from the video that are both:
    1. Stable (not much motion between current and previous frame)
    2. Contain at least one face

    Parameters:
    - video_path: path to video file
    - start_time and end_time: time window in seconds
    - time_gap: time between frames to sample
    - motion_threshold: how similar two frames must be to count as "stable"

    Returns:
    - A list of tuples: (timestamp, image as PIL.Image)
    """
    cap = cv2.VideoCapture(video_path)
    frames_with_faces = []
    prev_gray = None
    current_time = start_time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)

    while current_time < end_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for motion comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stable = True

        if prev_gray is not None:
            similarity = ssim(prev_gray, gray)
            if similarity < motion_threshold:
                stable = False
        prev_gray = gray

        if not stable:
            current_time += time_gap
            continue

        # Detect faces
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        boxes, _ = mtcnn.detect(pil_img)

        if boxes is not None and len(boxes) > 0:
            frames_with_faces.append((current_time, pil_img))

        current_time += time_gap

    cap.release()
    return frames_with_faces

# -------------------------- UI Logic --------------------------

video_path = None
option = st.radio("Select video source:", ["YouTube URL", "Upload Video"])

if option == "YouTube URL":
    video_url = st.text_input("Paste a YouTube video URL:")
    if video_url:
        with st.spinner("Downloading video..."):
            try:
                video_path = download_youtube_video(video_url)
                st.success("Video downloaded successfully")
            except Exception as e:
                st.error(f"Download error: {e}")

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        video_path = save_uploaded_video(uploaded_file)
        st.success("Video uploaded successfully")

# -------------------------- Frame Extraction --------------------------

if video_path:
    st.video(video_path)

    total_frames, fps, duration = load_video_metadata(video_path)
    st.info(f"Duration: {duration:.2f} seconds | FPS: {fps:.2f} | Total Frames: {total_frames}")

    time_range = st.slider("Select time range (seconds)", 0.0, duration, (0.0, duration), step=0.1)
    time_gap = st.number_input("Frame interval (seconds)", min_value=0.01, value=0.5, step=0.01)

    if st.button("Extract Frames (Faces + Stable Only)"):
        with st.spinner("Processing video..."):
            frames = extract_frames_with_faces_and_motion_filter(
                video_path=video_path,
                start_time=time_range[0],
                end_time=time_range[1],
                time_gap=time_gap,
                motion_threshold=0.80
            )
            st.success(f"Extracted {len(frames)} stable frames with faces")

            for t, img in frames:
                st.image(img, caption=f"Time: {t:.2f} seconds", use_container_width=True)
