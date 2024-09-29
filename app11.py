import streamlit as st
import openai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time
import googlemaps
from streamlit_folium import folium_static
import folium
from streamlit_js_eval import get_geolocation
import google.generativeai as genai

# Set up OpenAI API key
openai.api_key = "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"
gmaps = googlemaps.Client(key="AIzaSyBZ54CrwbNjBiKKs-4NydriYQTp0yEGFlM")
# Load the model
model = load_model('weights.h5')

genai.configure(api_key="AIzaSyA8CHnU_1P-UMjwR9bK9Fn77zmymPNXC5Y")


def get_chatbot_response(user_input):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    prompt = f"""
    You are a medical chatbot specializing in cancer treatment, diagnosis, and prevention. 
    Provide concise answers, but elaborate when necessary for complex topics. 
    Always prioritize accurate medical information and encourage users to consult healthcare professionals for personalized advice.

    User query: {user_input}
    """
    response = chat.send_message(user_input)
    return response.text

def get_nearby_hospitals(lat, lng):
    # Search for nearby cancer hospitals
    places_result = gmaps.places_nearby(
        location=(lat, lng),
        radius=10000,  # 10km in meters
        keyword='cancer hospital'
    )
    
    hospitals = places_result.get('results', [])[:10]  # Limit to top 10 results
    
    # Get additional details for each hospital
    for hospital in hospitals:
        place_id = hospital['place_id']
        details = gmaps.place(place_id, fields=['formatted_phone_number', 'website'])
        hospital['phone_number'] = details['result'].get('formatted_phone_number', 'N/A')
        hospital['website'] = details['result'].get('website', 'N/A')
    
    return hospitals

# Function to get a readable location name
def get_location_name(lat, lng):
    result = gmaps.reverse_geocode((lat, lng))
    if result:
        # Try to get the most specific address component
        for component in result[0]['address_components']:
            if 'sublocality' in component['types']:
                return component['long_name']
        # If no sublocality, try to get the locality
        for component in result[0]['address_components']:
            if 'locality' in component['types']:
                return component['long_name']
        # If no specific component found, return the formatted address
        return result[0]['formatted_address']
    return "Unknown location"

# Streamlit app layout
st.title("Image Classification with TensorFlow, OpenAI Chatbot, and Google Maps")
st.write("Upload 1 to 3 images to classify them as malignant or not.")

# Chatbot Interface
st.subheader("Chatbot")
user_input = st.text_input("You:", "")

if user_input:
    response = get_chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=200, max_chars=None, key=None)

# File uploader for image upload
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Ensure that at least one image is uploaded and a maximum of three images
if uploaded_files:
    if len(uploaded_files) < 1 or len(uploaded_files) > 3:
        st.warning("Please upload between 1 to 3 images.")
    else:
        for uploaded_file in uploaded_files:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write("")
            
            # Preprocess the image
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)

            # Make predictions
            prediction = model.predict(processed_image)
            class_label = "Malignant" if prediction[0] > 0.5 else "Not Malignant"  # Threshold for binary classification

            # Display the result
            st.write(f"Prediction: {class_label}")
else:
    st.warning("Please upload at least one image.")

# Google Maps Integration
st.subheader("Find Nearby Cancer Hospitals")

# Get user's location
loc = get_geolocation()

if loc:
    lat = loc['coords']['latitude']
    lng = loc['coords']['longitude']
    location_name = get_location_name(lat, lng)
    
    # Display location information using Streamlit components
    st.write(f"Your location: {location_name}")
    
    hospitals = get_nearby_hospitals(lat, lng)
    if hospitals:
        st.write(f"Found {len(hospitals)} cancer hospitals within 10km radius.")
        
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Nearest 10 Hospitals")
            for i, hospital in enumerate(hospitals, 1):
                st.write(f"{i}. {hospital['name']}")
                st.write(f"   Phone: {hospital['phone_number']}")
                if hospital['website'] != 'N/A':
                    st.write(f"   Website: [{hospital['website']}]({hospital['website']})")
                st.write("---")
        
        with col2:
            # Create a map centered on the user's location
            m = folium.Map(location=[lat, lng], zoom_start=12)
            
            # Add a marker for the user's location
            folium.Marker(
                location=[lat, lng],
                popup=f"Your Location: {location_name}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add markers for each hospital
            for hospital in hospitals:
                folium.Marker(
                    location=[hospital['geometry']['location']['lat'], hospital['geometry']['location']['lng']],
                    popup=hospital['name'],
                    tooltip=hospital['name']
                ).add_to(m)
            
            # Display the map
            folium_static(m)
    else:
        st.warning("No cancer hospitals found within 10km radius.")
else:
    st.warning("Unable to get your location. Please make sure you've granted location access to this site.")