import streamlit as st
import joblib
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
import re
import os
import random  # Import random to generate confidence values

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enhanced hate speech patterns
HATE_PATTERNS = {
    'profanity': [
        r'\b(fuck|shit|damn|bitch|ass|dick|pussy|cunt)\b',
        r'\b(asshole|motherfuck|bastard|whore|slut)\b'
    ],
    'slurs': [
        r'\b(nigger|faggot|retard|paki|chink)\b',
        r'\b(wetback|beaner|towelhead|raghead)\b'
    ],
    'threats': [
        r'\b(kill|murder|die|hang|shoot|slaughter|beat)\b',
        r'\b(threat|attack|hurt|harm|destroy|eliminate)\b'
    ],
    'discrimination': [
        r'\b(racist|sexist|nazi|fascist|bigot)\b',
        r'\b(hate|hating|hatred)\b'
    ],
    'offensive': [
        r'\b(stfu|gtfo|fck|fuk|fk|stfu|wtf|af)\b',
        r'[!\?]{2,}',  # Multiple exclamation/question marks
    ]
}

def check_hate_patterns(text):
    """Enhanced hate speech pattern checking"""
    text_lower = text.lower()
    
    # Check each category of patterns
    for category, patterns in HATE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True, 1.0  # Return True with high confidence
    
    return False, 0.0

def preprocess_text(text):
    """Enhanced text preprocessing"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize common substitutions
        text = text.replace('0', 'o')
        text = text.replace('1', 'i')
        text = text.replace('3', 'e')
        text = text.replace('4', 'a')
        text = text.replace('5', 's')
        text = text.replace('$', 's')
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text

def predict_text(text, model, tokenizer, label_encoder):
    try:
        # First check for obvious hate patterns
        is_hate, pattern_confidence = check_hate_patterns(text)
        if is_hate:
            return "yes", pattern_confidence, text
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize text
        encoding = tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='tf',
            return_attention_mask=True
        )
        
        # Make prediction
        with tf.device('/cpu:0'):
            outputs = model.predict([encoding['input_ids'], encoding['attention_mask']])
            logits = outputs.logits
        
        # Get prediction probabilities
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        predicted_label_id = np.argmax(probabilities)
        confidence = probabilities[predicted_label_id]
        
        # Get predicted label
        predicted_label = label_encoder.classes_[predicted_label_id]
        
        return predicted_label, confidence, processed_text
        
    except Exception as e:
        st.error(f"Error in Detection: {str(e)}")
        return None, 0.0, text

# Define hateful emojis and patterns
hateful_emojis = [u'😠', u'😡', u'🤬', u'🥵', u'🤢', u'🤮', u'👿', u'💩', u'👎', u'👎🏻', u'👎🏼', 
                  u'👎🏽', u'👎🏾', u'👎🏿', u'🖕', u'🖕🏻', u'🖕🏼', u'🖕🏽', u'🖕🏾', u'🖕🏿', 
                  u'👙', u'🩱', u'💦', u'🍌', u'🍑', u'🥊', u'🏴‍☠️']

def has_hateful_content(text):
    """Check for obvious hateful content"""
    # Check for hateful emojis
    if any(emoji in text for emoji in hateful_emojis):
        return True
    
    # Check for common hate speech patterns
    hate_patterns = [
        r'\b(hate|kill|die|murder)\b',
        r'\b(racist|sexist|bigot)\b',
        # Add more patterns as needed
    ]
    
    return any(re.search(pattern, text.lower()) for pattern in hate_patterns)

def text_classification_function():
    st.markdown('<div style="border: 3px solid pink; border-radius: 10px; padding: 10px;text-align: center;">Text Classification Task</div>', unsafe_allow_html=True)
    
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

        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_area("Enter text to classify:", height=100)

        if st.button("Predict"):
            if user_input:
                with st.spinner("Analyzing text..."):
                    predicted_label, confidence, processed_text = predict_text(
                        user_input, model, tokenizer, label_encoder
                    )

                    if predicted_label:
                        # Display processed text
                        st.markdown("### Processed Text:")
                        st.write(processed_text)
                        
                        # Display prediction with confidence
                        st.markdown("### Analysis Result:")
                        color = 'red' if predicted_label.lower() == 'yes' else 'green'
                        
                        st.markdown(
                            f'<div style="border: 2px solid {color}; border-radius: 10px; padding: 10px;">'
                            f'<p style="color: {color}; margin: 0;">Hate Speech Detected: {predicted_label}</p>'
                            f'<p style="margin: 0;">Confidence: {confidence:.2%}</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Additional context if hate speech detected
                        if predicted_label.lower() == 'yes':
                            st.warning("⚠️ This text contains potentially harmful content.")
            else:
                st.warning("Please enter some text.")
                
    except Exception as e:
        st.error(f"Error in text classification: {str(e)}")

if __name__ == "__main__":
    text_classification_function()
