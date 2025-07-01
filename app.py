import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import yt_dlp
from skimage.metrics import structural_similarity as ssim

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

def save_uploaded_video(uploaded_file_path: str) -> str:
    """
    Saves a user-uploaded video file to a temporary directory.
    - Returns the full file path where the video is saved.
    """
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, os.path.basename(uploaded_file_path))
    with open(uploaded_file_path, "rb") as src, open(video_path, "wb") as dst:
        dst.write(src.read())
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

# -------------------------- Example Usage --------------------------

if __name__ == "__main__":
    # You can use any one vedio_path according to your wish
    # Example: Download from YouTube
    video_path = download_youtube_video("https://youtube.com/your-video-link")

    # OR use a local file
    video_path = save_uploaded_video("/path/to/your/video.mp4")

    # Load metadata
    total_frames, fps, duration = load_video_metadata(video_path)
    print(f"Video Info → Duration: {duration:.2f}s | FPS: {fps:.2f} | Total Frames: {total_frames}")

    # Extract stable face-containing frames
    frames = extract_frames_with_faces_and_motion_filter(
        video_path=video_path,
        start_time=0,
        end_time=duration,
        time_gap=1
    )

    print(f"Extracted {len(frames)} stable frames with faces")

    # Save extracted frames as images
    for t, img in frames:
        save_path = f"frame_at_{t:.2f}_sec.jpg"
        img.save(save_path)
        print(f"Saved {save_path}")

