# Video Frame Extractor (Faces + Motion Filtering)

This project is a **Streamlit-based web app** that lets you extract **frames from a video** that meet two conditions:

1. The frame is **stable** (no fast motion or scene change)  
2. There is **at least one face** in the frame

You can either **upload a video** from your device or **paste a YouTube link**, and the app will:
- Download and read the video  
- Skip frames with fast motion  
- Detect faces using a deep learning model (MTCNN)  
- Show the final filtered frames with timestamp labels

---

## 🔧 Features

- Extract frames from YouTube or uploaded video  
- Uses motion filtering with SSIM to avoid blurry or fast-moving frames  
- Uses MTCNN face detection for accuracy  
- Returns clean and useful frames only

---

## 📁 Folder Structure

```
video-frame-extractor/
├── app.py                 # Main Streamlit app file
├── README.md              # This file
```

---

## 🚀 How to Run (Locally)

### 1. Install dependencies

```bash
pip install streamlit opencv-python-headless numpy Pillow facenet-pytorch yt-dlp scikit-image
```

### 2. Start the app

```bash
streamlit run app.py
```

---

This project is like a **smart photo taker from videos**.  
Imagine you’re watching a video and you want to save only the good shots where:
- The video isn’t shaking or moving fast  
- A face is clearly visible  

This tool watches the video for you, finds those moments, and saves them as pictures.  
You can use any YouTube video or your own.

---

## 📽️ Example Use Cases

- Pick clear face shots for face recognition datasets  
- Collect stable thumbnails from video lectures  
- Automatically filter out blurry or shaky scenes


