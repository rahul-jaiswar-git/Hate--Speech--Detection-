import streamlit as st
import cv2  # OpenCV for video processing
import speech_recognition as sr
import wave
import os
import time
import joblib
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
from pydub import AudioSegment  # Import pydub for audio extraction
import tempfile

# Specify the paths for all required files
model_path = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model\tf_model.h5'
config_path = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model\config.json'
tokenizer_config_path = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model\tokenizer_config.json'
vocab_path = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model\vocab.txt'
special_tokens_map_path = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model\special_tokens_map.json'

def delete_temporary_files():
    """Delete the temporary video and audio files."""
    temp_video_files = [f for f in os.listdir() if f.startswith("temp_video_") and f.endswith(".mp4")]
    temp_audio_files = [f for f in os.listdir() if f.startswith("temp_audio_") and f.endswith(".wav")]

    for file in temp_video_files + temp_audio_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

def load_model_and_tokenizer():
    try:
        # Load model and tokenizer
        model_directory = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model'
        model = TFBertForSequenceClassification.from_pretrained(model_directory)
        tokenizer = BertTokenizer.from_pretrained(model_directory)
        
        # Load model weights
        tf_model_filename = os.path.join(model_directory, 'tf_model.h5')
        model.load_weights(tf_model_filename)
        
        # Load label encoder
        label_encoder_filename = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\label_encoder.pkl'
        label_encoder = joblib.load(label_encoder_filename)
        
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def transcribe_video(video_file):
    """Transcribes the audio from a video file with improved accuracy."""
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, f'temp_video_{hash(video_file.name)}.mp4')
    temp_audio_path = os.path.join(temp_dir, f'temp_audio_{hash(video_file.name)}.wav')
    
    try:
        # Save the video content
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        # Check video duration
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        if duration > 240:  # 4 minutes limit
            return "Video duration exceeds 4 minutes limit."

        # Extract audio using pydub
        audio = AudioSegment.from_file(temp_video_path)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Normalize audio
        audio = audio.normalize()
        
        # Export as WAV
        audio.export(temp_audio_path, format="wav")

        # Initialize recognizer with improved settings
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15
        recognizer.dynamic_energy_ratio = 1.5
        recognizer.pause_threshold = 0.8

        # Transcribe audio
        with sr.AudioFile(temp_audio_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            
            try:
                text = recognizer.recognize_google(audio_data, language='en-IN')
                return text.strip() if text else "No speech detected in the video."
            except sr.UnknownValueError:
                return "Speech recognition could not understand the audio"
            except sr.RequestError as e:
                return f"Could not request results from speech recognition service; {e}"

    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

    finally:
        # Cleanup
        try:
            for file_path in [temp_video_path, temp_audio_path]:
                if os.path.exists(file_path):
                    time.sleep(0.1)
                    os.remove(file_path)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {str(e)}")

def video_classification_function():
    try:
        st.markdown('<div style="border: 3px solid pink; border-radius: 10px; padding: 10px;text-align: center;">Video Classification Task</div>', unsafe_allow_html=True)
        
        # Load model components
        model, tokenizer, label_encoder = load_model_and_tokenizer()
        
        if model is None or tokenizer is None or label_encoder is None:
            st.error("Failed to load model components")
            return

        # File uploader
        video_file = st.file_uploader("Upload a video file", type=["mp4"])

        if video_file is not None:
            # Display video
            st.video(video_file)

            # Process video and display transcription
            st.markdown("<h2 style='border-radius: 10px; border: 2px solid pink; padding: 10px;'>Transcription:</h2>", unsafe_allow_html=True)
            
            with st.spinner("Processing video..."):
                text = transcribe_video(video_file)
                
                if text and text not in ["Speech recognition could not understand the audio", "No speech detected in the video."]:
                    st.write(text)
                    
                    # Make prediction
                    with st.spinner("Analyzing content..."):
                        try:
                            # Tokenize text
                            encoding = tokenizer.encode_plus(
                                text,
                                add_special_tokens=True,
                                max_length=128,
                                padding='max_length',
                                truncation=True,
                                return_tensors='tf'
                            )
                            
                            # Predict
                            with tf.device('/cpu:0'):
                                outputs = model.predict([encoding['input_ids'], encoding['attention_mask']])
                                logits = outputs.logits
                            
                            # Get prediction
                            probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
                            predicted_label_id = np.argmax(probabilities)
                            predicted_label = label_encoder.classes_[predicted_label_id]
                            confidence = probabilities[predicted_label_id]
                            
                            # Display prediction
                            st.markdown("<h2 style='border-radius: 10px; padding: 10px;'>Prediction:</h2>", unsafe_allow_html=True)
                            color = "red" if predicted_label.lower() == "yes" else "green"
                            st.markdown(
                                f'<div style="border: 2px solid {color}; border-radius: 5px; padding: 10px;">'
                                f'<p style="color: {color}; margin: 0;">Prediction: {predicted_label}</p>'
                                f'<p style="margin: 0;">Confidence: {confidence:.2%}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                else:
                    st.error(text)  # Display the error message from transcription

    except Exception as e:
        st.error(f"Error in video classification: {str(e)}")

if __name__ == "__main__":
    video_classification_function()