import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

def run_app():
    # Load the trained machine learning model
    model = joblib.load('y_pred.pkl')

    # Streamlit app title and description
    st.title('House Price Prediction App')
    st.write('This app predicts house prices based on input features.')

    # User input for feature values
    st.sidebar.header('Input Features')
    # Example input fields, you should customize these based on your features
    input_features = {
        'housing_median_age': st.sidebar.slider('Age', min_value=6, max_value=52, value=1),
        'total_rooms': st.sidebar.slider('Rooms', min_value=0, max_value=39320, value=10),
        'total_bedrooms': st.sidebar.slider('Bedrooms', min_value=2, max_value=6210, value=10),
        'households': st.sidebar.slider('Households', min_value=2, max_value=5358, value=10),
        
        'longitude': st.sidebar.slider('longitude ', min_value=-125, max_value=-114, value=1),
        'latitude': st.sidebar.slider('latitude', min_value=33, max_value=42, value=1),
        'population': st.sidebar.slider('population', min_value=8, max_value=16305, value=100),
        'median_income': st.sidebar.slider('median_income', min_value=1, max_value=16, value=1),
        
        'INLAND': st.sidebar.slider('INLAND', min_value=0, max_value=1, value=1),
        'ISLAND': st.sidebar.slider('ISLAND', min_value=0, max_value=1, value=1),  
        'NEAR BAY': st.sidebar.slider('NEAR BAY', min_value=0, max_value=1, value=1),
        'NEAR OCEAN': st.sidebar.slider('NEAR OCEAN', min_value=0, max_value=1, value=1),
        # Add more feature sliders as needed
    }
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_features])

    # Display the input data
    st.subheader('Input Data')
    st.write(input_df)

    # Make predictions using the loaded model
    predicted_prices = model.predict(input_df)

    # Display the predicted prices
    st.subheader('Predicted Prices')
    st.write(predicted_prices)

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title='House Price Prediction App', page_icon=':house:')
    st.sidebar.title('Navigation')
    app_mode = st.sidebar.selectbox('Choose the app mode', ['App', 'About'])

    if app_mode == 'App':
        run_app()
    else:
        st.sidebar.subheader('About')
        st.sidebar.text('This app demonstrates a simple machine learning prediction.')
