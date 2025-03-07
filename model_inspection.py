import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import cv2
from transformers import TFBertForSequenceClassification, BertTokenizer
import joblib
import json

# Step 1: Load the pre-trained model
model = load_model(r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model\tf_model.h5')

# Step 2: Print the model summary to understand its architecture
model.summary()

# Step 3: Define preprocessing functions for text, audio, and image

def preprocess_text(text):
    # Implement your text preprocessing here
    # Example: Tokenize, pad sequences, etc.
    # For demonstration, let's assume it returns a numpy array
    return np.array([text])  # Replace with actual preprocessing

def preprocess_audio(audio_path):
    # Load and preprocess the audio file
    audio, sr = librosa.load(audio_path, sr=None)
    # You can extract features like MFCCs, spectrograms, etc.
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.expand_dims(mfccs, axis=0)  # Adjust shape for model input

def preprocess_image(image_path):
    # Load the image and preprocess it
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Assuming the model expects 224x224 images
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Step 4: Test the model with different inputs

# Test the model with a sample text
text_input = "This is a sample text for hate speech detection."
text_prediction = model.predict(preprocess_text(text_input))
print("Text Prediction:", text_prediction)

# Test the model with an audio file
audio_input_path = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\uploads\#Rowdies  #tvf  #gali video #comedy  #krk #rap.mp3'  # Update with your audio file path
audio_prediction = model.predict(preprocess_audio(audio_input_path))
print("Audio Prediction:", audio_prediction)

# Test the model with an image file
image_input_path = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\Hate.jpg'  # Update with your image file path
image_prediction = model.predict(preprocess_image(image_input_path))
print("Image Prediction:", image_prediction)

def inspect_model():
    try:
        # Model directory
        model_directory = r'C:\Users\rahul\Desktop\Hate_Speech_Detection_Hinglish-main\hate_speech_model'
        
        print("\nChecking model directory contents:")
        print(os.listdir(model_directory))
        
        print("\nLoading model and tokenizer...")
        model = TFBertForSequenceClassification.from_pretrained(model_directory)
        tokenizer = BertTokenizer.from_pretrained(model_directory)
        
        print("\nLoading model weights...")
        tf_model_filename = os.path.join(model_directory, 'tf_model.h5')
        model.load_weights(tf_model_filename)
        
        print("\nModel loaded successfully!")
        print("\nModel Summary:")
        model.summary()
        
        print("\nModel Configuration:")
        print(model.config)
        
        return "Model inspection completed successfully!"
        
    except Exception as e:
        print(f"\nError during model inspection: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None

if __name__ == "__main__":
    print("Starting model inspection...")
    result = inspect_model()
    if result:
        print("\nInspection completed successfully!")
    else:
        print("\nInspection failed!")
