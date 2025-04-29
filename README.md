is # Multi-Modal Hate Speech Detection System

A comprehensive system for detecting hate speech across multiple modalities (text, audio, video, and images).

## Project Structure

```
.
├── src/                    # Source code
│   ├── app.py              # Main application
│   ├── text_classification.py
│   ├── audio_classification.py
│   ├── video_classification.py
│   ├── image_classification.py
│   └── styles.css          # UI styling
│
├── models/                 # Model files
│   ├── hate_speech_model/  # BERT model files
│   └── label_encoder.pkl   # Label encoder
│
├── notebooks/              # Jupyter notebooks
│   ├── Saved Model Jupyter File.ipynb
│   ├── Saved_Model_Colab_File.ipynb
│   └── Hate Speech Detection Hinglish.ipynb
│
├── resources/              # Resource files
│   ├── Hate.jpg
│   ├── models.gif
│   ├── about us.gif
│   ├── audio.wav
│   └── testaudio
│
└── docs/                   # Documentation
    └── Software Engineering Projects - Computer Science at University of Bristol.mp4
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
cd src
streamlit run app.py
```

## Features

- Text Classification
- Audio Classification
- Video Classification
- Image Classification

## Requirements

See `requirements.txt` for the complete list of dependencies. "# Hate-Shield-AI-" 
