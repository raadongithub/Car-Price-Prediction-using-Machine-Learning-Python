import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Define custom CSS styles
custom_css = """
    <style>
    body {
        background-color: white;  /* White background for the entire app */
        color: black;             /* Black text color for the entire app */
        font-weight: bold;        /* Bold text for the entire app */
    }
    .stTextInput>div>div>input {
        color: black;  /* Black text color for input fields */
        width: 300px; /* Increase the width of the input boxes */
        height: 40px; /* Increase the height of the input boxes */
        font-size: 16px; /* Increase the font size of the input boxes */
    }
    .stButton>button {
        background-color: #4F4F4F;  /* Dark grey background for button */
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: white;
    }
    </style>
"""

st.set_page_config(
    page_title="Car Model Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

def load_model():
    # Load the pickled model
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def user_input_features():
    # User input fields
    st.sidebar.header("Input Features")
    model = st.sidebar.text_input("Model", "Corolla")
    manufacturer = st.sidebar.text_input("Manufacturer", "Toyota")
    fuel_type = st.sidebar.text_input("Fuel Type", "Petrol")
    vehicle_transmission = st.sidebar.text_input("Vehicle Transmission", "Manual")
    color = st.sidebar.text_input("Color", "White")
    body_type = st.sidebar.text_input("Body Type", "Sedan")
    registered_in = st.sidebar.text_input("Registered In", "Karachi")
    assembly = st.sidebar.text_input("Assembly", "Local")
    location = st.sidebar.text_input("Location", "Lahore")
    model_date = st.sidebar.number_input("Model Date", min_value=1990, max_value=2024, value=2015)
    mileage = st.sidebar.number_input("Mileage From Odometer", min_value=0, value=50000)
    engine_capacity = st.sidebar.number_input("Engine Capacity", min_value=500, max_value=5000, value=1800)

    # Create a dictionary of the input features
    data = {
        'model': model,
        'manufacturer': manufacturer,
        'fuelType': fuel_type,
        'vehicleTransmission': vehicle_transmission,
        'color': color,
        'bodyType': body_type,
        'RegisteredIn': registered_in,
        'Assembly': assembly,
        'location': location,
        'modelDate': model_date,
        'mileageFromOdometer': mileage,
        'EngineCapacity': engine_capacity
    }
    return pd.DataFrame([data])

def main():
    # Streamlit app main function
    st.title("Car Model Prediction")
    data_df = user_input_features()
    st.subheader("User Input Features")
    st.write(data_df)
    model = load_model()

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(data_df)
        st.subheader("Prediction")
        st.write(prediction)

if __name__ == "__main__":
    main()
