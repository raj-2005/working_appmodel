import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import googlemaps
import folium
from streamlit_folium import folium_static
from streamlit_js_eval import get_geolocation
import google.generativeai as genai

# Set up API keys (use secrets management in production)
gmaps = googlemaps.Client("AIzaSyBZ54CrwbNjBiKKs-4NydriYQTp0yEGFlM")
genai.configure("AIzaSyA8CHnU_1P-UMjwR9bK9Fn77zmymPNXC5Y")

# Load the model
model = load_model('weights.h5')

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
    places_result = gmaps.places_nearby(
        location=(lat, lng),
        radius=10000,  # 10km in meters
        keyword='cancer hospital'
    )
    
    hospitals = places_result.get('results', [])[:10]  # Limit to top 10 results
    for hospital in hospitals:
        place_id = hospital['place_id']
        details = gmaps.place(place_id, fields=['formatted_phone_number', 'website'])
        hospital['phone_number'] = details['result'].get('formatted_phone_number', 'N/A')
        hospital['website'] = details['result'].get('website', 'N/A')
    
    return hospitals

def get_location_name(lat, lng):
    result = gmaps.reverse_geocode((lat, lng))
    if result:
        for component in result[0]['address_components']:
            if 'sublocality' in component['types']:
                return component['long_name']
        for component in result[0]['address_components']:
            if 'locality' in component['types']:
                return component['long_name']
        return result[0]['formatted_address']
    return "Unknown location"

# Streamlit app layout
st.title("Cancer Detection and Information Chatbot")
st.write("Upload images to classify and interact with our chatbot.")

# Chatbot Interface
st.subheader("Chatbot")
user_input = st.text_input("You:", "")

if user_input:
    response = get_chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=200)

# File uploader for image upload
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) < 1 or len(uploaded_files) > 3:
        st.warning("Please upload between 1 to 3 images.")
    else:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            # Preprocess the image and make predictions
            image = Image.open(uploaded_file)
            processed_image = img_to_array(image.resize((224, 224))) / 255.0  # Adjust size as necessary
            processed_image = np.expand_dims(processed_image, axis=0)

            prediction = model.predict(processed_image)
            class_label = "Malignant" if prediction[0] > 0.5 else "Not Malignant"
            st.write(f"Prediction: {class_label}")

# Google Maps Integration
st.subheader("Find Nearby Cancer Hospitals")

loc = get_geolocation()
if loc:
    lat = loc['coords']['latitude']
    lng = loc['coords']['longitude']
    location_name = get_location_name(lat, lng)
    
    st.write(f"Your location: {location_name}")
    
    hospitals = get_nearby_hospitals(lat, lng)
    if hospitals:
        st.write(f"Found {len(hospitals)} cancer hospitals within 10km radius.")
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
            m = folium.Map(location=[lat, lng], zoom_start=12)
            folium.Marker(location=[lat, lng], popup=f"Your Location: {location_name}", icon=folium.Icon(color='red')).add_to(m)
            for hospital in hospitals:
                folium.Marker(location=[hospital['geometry']['location']['lat'], hospital['geometry']['location']['lng']], popup=hospital['name']).add_to(m)
            folium_static(m)
    else:
        st.warning("No cancer hospitals found within 10km radius.")
else:
    st.warning("Unable to get your location. Please ensure location access is granted.")
