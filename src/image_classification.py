import streamlit as st
import pytesseract as tess
from PIL import Image
import joblib
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
import re
import cv2
import os
from io import BytesIO

# Set Tesseract path
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def enhance_image(image):
    """Basic image enhancement for OCR"""
    try:
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
        
    except Exception as e:
        st.error(f"Error in image enhancement: {str(e)}")
        return np.array(image)

def extract_text(image):
    """Extract text from image"""
    try:
        # Basic enhancement
        enhanced_img = enhance_image(image)
        
        # Extract text using default configuration
        extracted_text = tess.image_to_string(enhanced_img, lang='eng')
        
        # Clean the extracted text
        cleaned_text = ' '.join(extracted_text.split())
        
        return cleaned_text if cleaned_text.strip() else "No text detected in image"
        
    except Exception as e:
        st.error(f"Error in text extraction: {str(e)}")
        return "Error extracting text from image"

def predict_text(text, model, tokenizer, label_encoder):
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
        
        # Make prediction
        outputs = model.predict([encoding['input_ids'], encoding['attention_mask']])
        logits = outputs.logits
        
        # Get prediction probabilities
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        predicted_label_id = np.argmax(probabilities)
        confidence = probabilities[predicted_label_id]
        
        # Get predicted label
        predicted_label = label_encoder.classes_[predicted_label_id]
        
        return predicted_label, confidence
        
    except Exception as e:
        st.error(f"Error in Detection: {str(e)}")
        return None, 0.0

def image_classification_function():
    st.markdown('<div style="border: 3px solid pink; border-radius: 10px; padding: 10px;text-align: center;">Image Classification Task</div>', unsafe_allow_html=True)
    
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

        # File uploader
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_image)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                with st.spinner("Processing image..."):
                    # Extract text from image
                    extracted_text = extract_text(image)
                    
                    # Display extracted text
                    st.markdown("### Extracted Text:")
                    st.write(extracted_text)
                    
                    if extracted_text and extracted_text != "No text detected in image":
                        # Make prediction
                        predicted_label, confidence = predict_text(
                            extracted_text, model, tokenizer, label_encoder
                        )
                        
                        if predicted_label:
                            # Display prediction
                            st.markdown("### Analysis Result:")
                            color = "red" if predicted_label.lower() == "yes" else "green"
                            st.markdown(
                                f'<div style="border: 2px solid {color}; border-radius: 10px; padding: 10px;">'
                                f'<p style="color: {color}; margin: 0;">Hate Speech Detected: {predicted_label}</p>'
                                f'<p style="margin: 0;">Confidence: {confidence:.2%}</p>'
                                '</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.warning("No text was detected in the image.")
                        
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                
    except Exception as e:
        st.error(f"Error in image classification: {str(e)}")

if __name__ == "__main__":
    image_classification_function()
