import os
import glob
import subprocess
import json
import pandas as pd
from transformers import pipeline
import whisper

# Load models once (can be optimized further)
sentiment_analyzer = None
whisper_model = None

def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    return sentiment_analyzer

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
    return whisper_model

def download_youtube_video(url, job_id):
    """Download YouTube video using yt-dlp"""
    video_path = f"job_{job_id}_video.mp4"
    result = subprocess.run(
        ["yt-dlp", "-f", "bv*+ba/b", "-o", video_path, url],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        raise Exception(f"Video download failed: {result.stderr}")
    
    # Check if video file exists (yt-dlp might add extension)
    if not os.path.exists(video_path):
        possible_files = glob.glob(f"job_{job_id}_video.*")
        if possible_files:
            video_path = possible_files[0]
        else:
            raise Exception("Video file not found after download.")
    
    return video_path

def extract_audio(video_path, job_id):
    """Extract audio from video using ffmpeg"""
    audio_path = f"job_{job_id}_audio.wav"
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Audio extraction failed: {result.stderr}")
    
    if not os.path.exists(audio_path):
        raise Exception("Audio file not found after extraction.")
    
    return audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    model = get_whisper_model()
    result = model.transcribe(audio_path)
    return result.get("segments", [])

def process_and_save(segments, audio_path, job_prefix):
    """Process segments for sentiment analysis and save results"""
    analyzer = get_sentiment_analyzer()
    
    results = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        
        # Get sentiment
        try:
            sentiment_result = analyzer(text)[0]
            label = sentiment_result["label"]
            score = sentiment_result["score"]
        except Exception as e:
            label = "ERROR"
            score = 0.0
        
        results.append({
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "text": text,
            "sentiment": label,
            "confidence": round(score, 4)
        })
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = f"{job_prefix}_sentiment.csv"
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = f"{job_prefix}_sentiment.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return csv_path, json_path

