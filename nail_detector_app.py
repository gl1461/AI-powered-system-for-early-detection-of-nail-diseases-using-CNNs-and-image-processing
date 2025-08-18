import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import json

# Load the trained model
model = load_model("naildisease_model.h5")

# Define class labels
class_labels = ['pitting', 'blue_finger', 'clubbing', 'Onychogryphosis', 'Healthy_Nail', 'Acral_Lentiginous_Melanoma']

# Set the page title
st.title('Nail Disease Detector')

# File uploader
uploaded_file = st.file_uploader("Upload a picture of the nail", type=["jpg", "jpeg", "png"])

# Simulate retrieving historical data (this would typically be a database or file system)
def retrieve_previous_results():
    try:
        with open('historical_results.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_new_result(new_result):
    try:
        with open('historical_results.json', 'r') as file:
            historical_results = json.load(file)
    except FileNotFoundError:
        historical_results = []

    historical_results.append(new_result)

    with open('historical_results.json', 'w') as file:
        json.dump(historical_results, file)

def calculate_similarity(current_features, previous_features):
    dot_product = np.dot(current_features, previous_features)
    norm_a = np.linalg.norm(current_features)
    norm_b = np.linalg.norm(previous_features)
    return dot_product / (norm_a * norm_b)

def preprocess_image(image):
    # Resize and preprocess the image
    image = image.resize((224, 224))  # Resize the image to (224, 224)
    img_array = img_to_array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if uploaded_file is not None:
    # Open and convert to RGB
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]

    # Show the prediction and confidence
    st.markdown(f"### Prediction: **{predicted_class}**")
    

    # Retrieve previous results for comparison
    historical_results = retrieve_previous_results()

    # Check if we have any historical results to compare
    if historical_results:
        last_result = historical_results[-1]  # Assume we compare with the last entry
        last_features = np.array(last_result['features'])

        similarity_score = calculate_similarity(prediction.flatten(), last_features)
        if similarity_score > 0.9:
            st.write("Your condition has not changed significantly.")
        elif similarity_score < 0.5:
            st.write("Your condition has worsened. Consider visiting a healthcare provider.")
        else:
            st.write("Your condition has improved. Keep following your treatment plan!")

    # Save the current result (prediction and features)
    new_result = {
        'date': str(np.datetime64('now')),  # Timestamp of the current result
        'condition': predicted_class,
        'features': prediction.flatten().tolist()  # Save prediction output as features
    }
    save_new_result(new_result)

    # Additional disease relations
    related_conditions = {
        'pitting': ['Psoriasis', 'Alopecia Areata', 'Eczema'],
        'blue_finger': ['Raynaud’s Disease', 'Peripheral Arterial Disease (PAD)', 'Cold Exposure'],
        'clubbing': ['Lung Diseases (COPD, lung cancer, cystic fibrosis)', 'Heart Disease (Low oxygen levels)', 'Inflammatory Bowel Disease (IBD)', 'Cirrhosis of the Liver'],
        'Onychogryphosis': ['Aging', 'Fungal Infections', 'Poor Circulation'],
        'Healthy_Nail': ['No health condition', 'Good hygiene', 'Proper nutrition (biotin, vitamins, and minerals)', 'Avoiding harsh chemicals', 'Regular trimming and moisturizing'],
        'Acral_Lentiginous_Melanoma': ['Melanoma', 'Non-Sun-Induced Skin Cancers', 'Immune Suppression (HIV, immunosuppressive drugs)']
    }

    if predicted_class in related_conditions:
        st.info(f"**Possible related condition(s):** {', '.join(related_conditions[predicted_class])}")
