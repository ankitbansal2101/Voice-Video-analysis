# Video & Audio Analysis Tool (Flask)

This project provides a web-based tool for analyzing YouTube videos with both **video analysis** (face detection, scene classification, color palette extraction) and **audio analysis** (transcription, sentiment analysis). The backend is built with Flask and features a unified UI that shows real-time progress for both analyses.

## Features

### Video Analysis
- Download YouTube videos or upload local video files
- Extract frames at custom FPS and time range
- Analyze faces (age, gender, emotion) with DeepFace
- Scene classification with CLIP
- Color palette extraction
- Downloadable results (video, summary CSV, frame analysis CSV)

### Audio Analysis
- Extract audio from video
- Transcribe audio using Whisper
- Sentiment analysis on transcribed text
- Downloadable results (sentiment CSV, sentiment JSON)

### UI Features
- Unified interface showing both video and audio progress
- Real-time progress bars for each analysis
- Step-by-step progress indicators
- Support for YouTube URLs and file uploads

## Parameters Explained
- **Start Time (s):** The time (in seconds) from which to begin extracting frames from the video.
- **End Time (s):** The time (in seconds) at which to stop extracting frames from the video.
- **FPS (Frames Per Second):** The number of frames to extract per second from the selected video segment. Higher FPS means more frames and more detailed analysis, but also longer processing time.

## Setup

### 1. Clone the repository
```
git clone https://github.com/ankitbansal2101/Voice-Viode-analysis
cd Voice-Viode-analysis
```

### 2. Install dependencies
It is recommended to use a virtual environment (venv or conda).

```
pip install -r requirements.txt
```

### 3. Install system dependencies
- **ffmpeg** (for frame extraction and audio processing)
- **yt-dlp** (for YouTube download, already in requirements.txt)

On macOS (with Homebrew):
```
brew install ffmpeg
```
On Ubuntu:
```
sudo apt-get install ffmpeg
```

### 4. Run the Flask server
```
python app.py
```

The server will start at [http://localhost:5000](http://localhost:5000).

## Usage
1. Open your browser and go to [http://localhost:5000](http://localhost:5000)
2. Enter a YouTube URL **OR** upload a video file
3. Set the start time, end time, and FPS parameters
4. Click "Start Analysis" and watch the progress bars for both video and audio analysis
5. Download the results when both analyses are complete

## Project Structure
```
.
├── app.py                 # Main Flask application (all functionality)
├── templates/
│   └── index.html        # Unified UI template
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Output Files
- **Video file**: The downloaded/uploaded video
- **Summary CSV**: Video analysis summary (avg age, dominant emotion, scene, etc.)
- **Frame Analysis CSV**: Detailed frame-by-frame analysis
- **Sentiment CSV**: Audio transcription with sentiment analysis
- **Sentiment JSON**: Same data in JSON format

## Notes
- The first run may take longer as models are downloaded (Whisper, CLIP, DeepFace, sentiment models)
- For large videos or high FPS, processing may take several minutes
- Both video and audio analyses run in parallel after the video is downloaded/uploaded
- All results are saved in the server's working directory
- Generated files (videos, frames, CSVs) are excluded from git via .gitignore

## Troubleshooting
- If you see errors about numpy or binary incompatibility, ensure you are using numpy 1.x (see requirements.txt)
- If video download fails, check that yt-dlp is installed and up to date: `pip install --upgrade yt-dlp`
- If audio transcription is slow, consider using a smaller Whisper model (change "base" to "tiny" in app.py)
- For any issues, check the terminal output for error messages

## License
MIT
