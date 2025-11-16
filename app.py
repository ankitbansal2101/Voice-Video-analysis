import os
import glob
import threading
import uuid
import time
import subprocess
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import numpy as np
import pandas as pd
from PIL import Image
from deepface import DeepFace
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import torch
from audio_analysis import download_youtube_video, extract_audio, transcribe_audio, process_and_save

app = Flask(__name__)

# In-memory job store
jobs = {}

# Helper: Video Analysis function
def run_video_analysis(job_id, video_path, start_time, end_time, fps):
    """Run video analysis (frames, faces, scenes, palettes)"""
    try:
        jobs[job_id]['video']['status'] = 'extracting_frames'
        jobs[job_id]['video']['progress'] = 0.1
        os.makedirs(f"frames_{job_id}", exist_ok=True)
        totaltime = end_time - start_time
        
        # Extract frames with error handling
        result = subprocess.run(
            ["ffmpeg", "-ss", str(start_time), "-t", str(totaltime), "-i", video_path, 
             "-vf", f"fps={fps}", f"frames_{job_id}/frame_%03d.jpg", "-hide_banner", "-loglevel", "error"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Frame extraction failed: {result.stderr}")
        frame_paths = sorted(glob.glob(f"frames_{job_id}/frame_*.jpg"))

        jobs[job_id]['video']['status'] = 'deepface_analysis'
        jobs[job_id]['video']['progress'] = 0.2
        frame_results = []
        for i, frame in enumerate(frame_paths):
            try:
                analysis = DeepFace.analyze(frame, actions=["age", "gender", "emotion"], enforce_detection=False)
                frame_results.append({
                    "frame": frame,
                    "age": analysis[0]["age"],
                    "gender": analysis[0]["gender"],
                    "emotion": analysis[0]["dominant_emotion"]
                })
            except Exception:
                frame_results.append({"frame": frame, "age": None, "gender": None, "emotion": None})
            jobs[job_id]['video']['progress'] = 0.2 + 0.3 * (i+1)/len(frame_paths) if frame_paths else 0.5

        jobs[job_id]['video']['status'] = 'clip_classification'
        jobs[job_id]['video']['progress'] = 0.5
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        scene_labels = ["bedroom", "kitchen", "street", "mirror selfie", "dance", "food", "product demo"]
        def classify_scene(image_path):
            image = Image.open(image_path).convert("RGB")
            inputs = clip_processor(text=scene_labels, images=image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy().flatten()
            return scene_labels[np.argmax(probs)]
        for r in frame_results:
            r["scene"] = classify_scene(r["frame"])
        jobs[job_id]['video']['progress'] = 0.7

        jobs[job_id]['video']['status'] = 'palette_extraction'
        def extract_palette(img_path, k=5):
            img = Image.open(img_path).convert("RGB")
            data = np.array(img).reshape(-1, 3)
            kmeans = KMeans(n_clusters=k, n_init="auto").fit(data)
            return kmeans.cluster_centers_.astype(int)
        for r in frame_results:
            try:
                r["palette"] = extract_palette(r["frame"]).tolist()
            except:
                r["palette"] = None
        jobs[job_id]['video']['progress'] = 0.85

        jobs[job_id]['video']['status'] = 'summary'
        df = pd.DataFrame(frame_results) if frame_results else pd.DataFrame()
        summary = {
            "avg_age": float(df["age"].dropna().mean()) if "age" in df.columns and df["age"].notna().sum() > 0 else None,
            "gender_mode": df["gender"].mode()[0] if "gender" in df.columns and df["gender"].notna().sum() > 0 else None,
            "dominant_emotion": df["emotion"].mode()[0] if "emotion" in df.columns and df["emotion"].notna().sum() > 0 else None,
            "dominant_scene": df["scene"].mode()[0] if "scene" in df.columns and df["scene"].notna().sum() > 0 else None,
            "sample_palette": df["palette"].iloc[0] if "palette" in df.columns and df["palette"].notna().sum() > 0 else None,
        }
        if not df.empty:
            df.to_csv(f"frame_analysis_{job_id}.csv", index=False)
        else:
            pd.DataFrame(columns=["frame", "age", "gender", "emotion", "scene", "palette"]).to_csv(f"frame_analysis_{job_id}.csv", index=False)
        pd.DataFrame([summary]).to_csv(f"summary_{job_id}.csv", index=False)
        jobs[job_id]['video']['progress'] = 1.0
        jobs[job_id]['video']['status'] = 'done'
        jobs[job_id]['files']['summary'] = f"summary_{job_id}.csv"
        jobs[job_id]['files']['frame_analysis'] = f"frame_analysis_{job_id}.csv"
    except Exception as e:
        jobs[job_id]['video']['status'] = 'error'
        jobs[job_id]['video']['error'] = str(e)

# Helper: Audio Analysis function
def run_audio_analysis(job_id, video_path):
    """Run audio analysis (transcription, sentiment)"""
    try:
        jobs[job_id]['audio']['status'] = 'extracting'
        jobs[job_id]['audio']['progress'] = 0.1
        print(f"[Job {job_id}] Extracting audio...")
        audio_path = extract_audio(video_path, job_id)
        
        jobs[job_id]['audio']['status'] = 'transcribing'
        jobs[job_id]['audio']['progress'] = 0.3
        print(f"[Job {job_id}] Transcribing audio...")
        segments = transcribe_audio(audio_path)
        
        jobs[job_id]['audio']['status'] = 'analyzing'
        jobs[job_id]['audio']['progress'] = 0.7
        print(f"[Job {job_id}] Processing and analyzing segments...")
        process_and_save(segments, audio_path, f"job_{job_id}")
        
        jobs[job_id]['audio']['progress'] = 1.0
        jobs[job_id]['audio']['status'] = 'done'
        jobs[job_id]['files']['sentiment_csv'] = f"job_{job_id}_sentiment.csv"
        jobs[job_id]['files']['sentiment_json'] = f"job_{job_id}_sentiment.json"
        print(f"[Job {job_id}] Audio analysis complete.")
    except Exception as e:
        print(f"[Job {job_id}] Audio analysis error: {e}")
        jobs[job_id]['audio']['status'] = 'error'
        jobs[job_id]['audio']['error'] = str(e)

# Main analysis function
def run_combined_analysis(job_id, video_url, start_time, end_time, fps, uploaded_file=None):
    """Run both video and audio analysis"""
    try:
        # Get video file (either download or use uploaded)
        if uploaded_file:
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['video']['status'] = 'processing'
            jobs[job_id]['audio']['status'] = 'waiting'
            jobs[job_id]['video']['progress'] = 0.0
            jobs[job_id]['audio']['progress'] = 0.0
            print(f"[Job {job_id}] Using uploaded file: {uploaded_file}")
            video_path = uploaded_file
        else:
            # Download video (shared by both analyses)
            jobs[job_id]['status'] = 'downloading'
            jobs[job_id]['video']['status'] = 'downloading'
            jobs[job_id]['audio']['status'] = 'waiting'
            jobs[job_id]['video']['progress'] = 0.0
            jobs[job_id]['audio']['progress'] = 0.0
            
            print(f"[Job {job_id}] Downloading video from URL: {video_url}")
            video_path = download_youtube_video(video_url, job_id)
        
        # Update job files
        jobs[job_id]['files'] = {'video': video_path}
        
        # Start both analyses in parallel
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['video']['status'] = 'processing'
        jobs[job_id]['audio']['status'] = 'processing'
        
        # Run video and audio analysis in separate threads
        video_thread = threading.Thread(target=run_video_analysis, args=(job_id, video_path, start_time, end_time, fps))
        audio_thread = threading.Thread(target=run_audio_analysis, args=(job_id, video_path))
        
        video_thread.start()
        audio_thread.start()
        
        # Wait for both to complete
        video_thread.join()
        audio_thread.join()
        
        # Check if both completed successfully
        video_done = jobs[job_id]['video']['status'] == 'done'
        audio_done = jobs[job_id]['audio']['status'] == 'done'
        
        if video_done and audio_done:
            jobs[job_id]['status'] = 'done'
        elif jobs[job_id]['video']['status'] == 'error' or jobs[job_id]['audio']['status'] == 'error':
            jobs[job_id]['status'] = 'error'
            if jobs[job_id]['video']['status'] == 'error':
                jobs[job_id]['error'] = f"Video: {jobs[job_id]['video'].get('error', 'Unknown error')}"
            if jobs[job_id]['audio']['status'] == 'error':
                audio_err = jobs[job_id]['audio'].get('error', 'Unknown error')
                if 'error' in jobs[job_id]:
                    jobs[job_id]['error'] += f" | Audio: {audio_err}"
                else:
                    jobs[job_id]['error'] = f"Audio: {audio_err}"
        
    except Exception as e:
        print(f"[Job {job_id}] Error: {e}")
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        if 'video' in jobs[job_id]:
            jobs[job_id]['video']['status'] = 'error'
        if 'audio' in jobs[job_id]:
            jobs[job_id]['audio']['status'] = 'error'

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        data = request.form
    video_url = data.get('url')
    start_time = int(data.get('start_time', 10))
    end_time = int(data.get('end_time', 100))
    fps = int(data.get('fps', 1))
    job_id = str(uuid.uuid4())
    
    # Initialize job with separate video and audio tracking
    jobs[job_id] = {
        'status': 'queued',
        'video': {'status': 'queued', 'progress': 0.0, 'error': None},
        'audio': {'status': 'queued', 'progress': 0.0, 'error': None},
        'files': {},
        'error': None
    }
    
    thread = threading.Thread(target=run_combined_analysis, args=(job_id, video_url, start_time, end_time, fps, None))
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    job_id = str(uuid.uuid4())
    filename = f"job_{job_id}_video.mp4"
    file.save(filename)
    
    # Get parameters from form or use defaults
    start_time = int(request.form.get('start_time', 10))
    end_time = int(request.form.get('end_time', 100))
    fps = int(request.form.get('fps', 1))
    
    # Initialize job with separate video and audio tracking
    jobs[job_id] = {
        'status': 'uploaded',
        'video': {'status': 'uploaded', 'progress': 0.0, 'error': None},
        'audio': {'status': 'queued', 'progress': 0.0, 'error': None},
        'files': {},
        'error': None
    }
    
    thread = threading.Thread(target=run_combined_analysis, args=(job_id, None, start_time, end_time, fps, filename))
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/progress/<job_id>')
def progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job.get('status', 'unknown'),
        'video': job.get('video', {}),
        'audio': job.get('audio', {}),
        'error': job.get('error')
    })

@app.route('/status/<job_id>')
def job_status(job_id):
    """Legacy endpoint for compatibility with original audio analysis UI"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not found'}), 404
    
    # Map combined status to simple status for compatibility
    status = job.get('status', 'unknown')
    if status == 'done':
        return jsonify({
            'status': 'done',
            'progress': 100,
            'files': job.get('files', {})
        })
    elif status == 'error':
        return jsonify({
            'status': f'error: {job.get("error", "Unknown error")}',
            'progress': -1
        })
    else:
        # Return the main status
        return jsonify({
            'status': status,
            'progress': 0
        })

@app.route('/download/<job_id>/<filetype>')
def download(job_id, filetype):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    file_map = job.get('files', {})
    
    # Map filetype to actual filename (including legacy names)
    filetype_map = {
        'video': 'video',
        'summary': 'summary',
        'frame_analysis': 'frame_analysis',
        'sentiment_csv': 'sentiment_csv',
        'sentiment_json': 'sentiment_json',
        'csv': 'sentiment_csv',  # Legacy support
        'json': 'sentiment_json'  # Legacy support
    }
    
    actual_filetype = filetype_map.get(filetype, filetype)
    
    if actual_filetype not in file_map:
        return jsonify({'error': 'File not ready or invalid file type'}), 404
    
    filename = file_map[actual_filetype]
    if not os.path.exists(filename):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
