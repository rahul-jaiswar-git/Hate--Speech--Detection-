import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import mediainfo
import io
import tempfile
import os
import joblib
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
import pytesseract
import soundfile as sf
import librosa
import cv2
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Explicitly set the path to ffmpeg.exe
AudioSegment.converter = r'C:\Users\rahul\Downloads\ffmpeg-7.1-full_build\ffmpeg-7.1-full_build\bin\ffmpeg.exe'

# Set paths
model_directory = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model'
tf_model_filename = os.path.join(model_directory, 'tf_model.h5')
label_encoder_filename = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\label_encoder.pkl'

def load_model_and_tokenizer():
    try:
        # Load model and tokenizer
        model = TFBertForSequenceClassification.from_pretrained(model_directory)
        tokenizer = BertTokenizer.from_pretrained(model_directory)
        
        # Load model weights
        model.load_weights(tf_model_filename)
        
        # Load label encoder
        label_encoder = joblib.load(label_encoder_filename)
        
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_text(text, model, tokenizer, label_encoder):
    try:
        # Tokenize text
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # Make prediction
        with tf.device('/cpu:0'):
            outputs = model.predict([inputs['input_ids'], inputs['attention_mask']])
            logits = outputs.logits
        
        # Get prediction probabilities
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        predicted_label_id = np.argmax(probabilities)
        confidence = probabilities[predicted_label_id]
        
        # Get predicted label
        predicted_label = label_encoder.classes_[predicted_label_id]
        
        return predicted_label, confidence
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, 0.0

def process_audio(audio_file):
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    temp_path = os.path.join(temp_dir, 'temp_audio.wav')
    
    try:
        # Save the uploaded file to the temporary path
        if audio_file.name.endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_file)
            audio.export(temp_path, format='wav')
        else:
            # For WAV files, directly save
            with open(temp_path, 'wb') as f:
                f.write(audio_file.getbuffer())

        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Read audio file
        with sr.AudioFile(temp_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            # Record audio
            audio_data = recognizer.record(source)
        
        # Convert speech to text
        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return "Speech not recognized"
    except sr.RequestError as e:
        return f"Error in speech recognition service: {str(e)}"
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None
    finally:
        # Clean up: Close any open handles and remove temporary files
        try:
            if os.path.exists(temp_path):
                # Wait a bit before trying to delete
                time.sleep(0.1)
                os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {str(e)}")

def audio_classification_function(audio_file=None):
    try:
        st.markdown('<div style="border: 3px solid pink; border-radius: 10px; padding: 10px;text-align: center;">Audio Classification Task</div>', unsafe_allow_html=True)
        
        # Load model and tokenizer
        model, tokenizer, label_encoder = load_model_and_tokenizer()
        
        if model is None or tokenizer is None or label_encoder is None:
            st.error("Failed to load model components")
            return None
        
        # Use provided audio file or file uploader
        if audio_file is None:
            audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        
        if audio_file is not None:
            # Display audio player
            st.audio(audio_file)
            
            with st.spinner("Processing audio..."):
                # Convert speech to text
                text = process_audio(audio_file)
                
                if text and text != "Speech not recognized":
                    # Display transcribed text
                    st.markdown("### Transcribed Text:")
                    st.write(text)
                    
                    # Make prediction
                    predicted_label, confidence = predict_text(text, model, tokenizer, label_encoder)
                    
                    if predicted_label:
                        # Display prediction with confidence
                        st.markdown("### Prediction:")
                        color = "red" if predicted_label.lower() == "yes" else "green"
                        st.markdown(
                            f'<div style="border: 2px solid {color}; border-radius: 5px; padding: 10px;">'
                            f'<p style="color: {color}; margin: 0;">Prediction: {predicted_label}</p>'
                            f'<p style="margin: 0;">Confidence: {confidence:.2%}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        return predicted_label
                else:
                    st.error("Could not recognize speech in the audio file")
                    return None
        return None
        
    except Exception as e:
        st.error(f"Error in audio classification: {str(e)}")
        return None

def transcribe_video(video_file):
    """Transcribes the audio from a video file."""
    
    # Create temporary files with unique names
    temp_video_file_path = os.path.join(os.getcwd(), "temp_video_" + str(hash(video_file.name)) + ".mp4")
    temp_wav_file_path = os.path.join(os.getcwd(), "temp_audio_" + str(hash(video_file.name)) + ".wav")

    try:
        # Save the video content to a temporary file
        with open(temp_video_file_path, "wb") as f:
            f.write(video_file.read())

        # Use OpenCV to read the video
        cap = cv2.VideoCapture(temp_video_file_path)

        # Check video duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        if duration > 240:  # 4 minutes = 240 seconds
            return st.error("The uploaded video must be less than or equal to 4 minutes in duration.")

        # Extract audio using pydub
        audio_segment = AudioSegment.from_file(temp_video_file_path)  # Load video file as audio
        audio_segment.export(temp_wav_file_path, format="wav")  # Export as WAV file

        # Initialize recognizer
        r = sr.Recognizer()

        # Transcribe audio
        with sr.AudioFile(temp_wav_file_path) as source:
            data = r.record(source)

        # Convert speech to text
        text = r.recognize_google(data)

        return text

    except Exception as e:
        print(f"Error during transcription: {e}")
        return st.error("An error occurred while transcribing the video. Please try again.")

    finally:
        # Ensure temporary files are deleted even if exceptions occur
        if cap.isOpened():
            cap.release()  # Release the video capture object
        # Add a small delay before deleting the file to ensure it's not in use
        time.sleep(0.1)  # Wait for a short time
        if os.path.exists(temp_video_file_path):
            try:
                os.unlink(temp_video_file_path)
            except PermissionError as e:
                print(f"Error deleting file {temp_video_file_path}: {e}")
                # Optionally, you can retry deletion after a short delay
                time.sleep(0.5)
                try:
                    os.unlink(temp_video_file_path)
                except Exception as e:
                    print(f"Failed to delete file after retry: {e}")

if __name__ == "__main__":
    audio_classification_function()
