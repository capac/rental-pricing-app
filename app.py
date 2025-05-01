import streamlit as st
import joblib
import numpy as np
import pandas as pd


@st.cache_resource
def load_model():
    return joblib.load('model.pkl')


model = load_model()

st.title('London Short-Term Rental Price Estimator')
st.write(
    'Use this tool to estimate the nightly price '
    'of your rental property based on its characteristics '
    'and location in London.'
    )

property_type = st.selectbox(
    'Property Type',
    ['Entire rental unit', 'Entire townhouse', 'Entire condo',
     'Entire serviced apartment', 'Entire guest suite',
     'Entire loft', 'Entire home', 'Entire guesthouse',
     'Room in aparthotel', 'Entire place', 'Camper/RV',
     'Tiny home', 'Boat', 'Entire bungalow']
    )
bedrooms = st.number_input(
    'Number of bedrooms', min_value=0, max_value=20, value=1, step=1
    )
bathrooms = st.number_input(
    'Number of bathrooms', min_value=0, max_value=20, value=1, step=1
    )
minimum_nights = st.number_input(
    'Minimum nights stay', min_value=1, max_value=999, value=3, step=1
    )
borough = st.selectbox(
    'London borough',
    ['Lambeth', 'Kensington and Chelsea', 'Brent', 'Westminster',
     'Hammersmith and Fulham', 'Islington', 'Hackney', 'Camden',
     'Wandsworth', 'Southwark', 'Haringey', 'Hounslow', 'Tower Hamlets',
     'Barnet', 'Richmond upon Thames', 'Enfield', 'Newham',
     'Kingston upon Thames', 'Lewisham', 'Harrow', 'Croydon',
     'Greenwich', 'Merton', 'Ealing', 'Bromley', 'City of London',
     'Waltham Forest', 'Hillingdon', 'Bexley', 'Havering', 'Redbridge',
     'Barking and Dagenham']
     )
distance_to_station = st.slider(
    'Distance to nearest station (km)',
    min_value=0.0, max_value=20.0, value=0.5, step=0.1
    )

amenity_1 = st.selectbox(
    'Nearby amenity 1',
    ['Grocery Store', 'Restaurant', 'Cafe', 'Nightlife', 'Retail',
     'Fitness', 'Wellness', 'Entertainment', 'Cultural', 'Outdoor',
     'Transport', 'Healthcare', 'Services', 'Organization',
     'Education', 'Religion', 'Home Improvement']
    )
amenity_2 = st.selectbox(
    'Nearby amenity 2',
    ['Grocery Store', 'Restaurant', 'Cafe', 'Nightlife', 'Retail',
     'Fitness', 'Wellness', 'Entertainment', 'Cultural', 'Outdoor',
     'Transport', 'Healthcare', 'Services', 'Organization',
     'Education', 'Religion', 'Home Improvement']
    )
amenity_3 = st.selectbox(
    'Nearby amenity 3',
    ['Grocery Store', 'Restaurant', 'Cafe', 'Nightlife', 'Retail',
     'Fitness', 'Wellness', 'Entertainment', 'Cultural', 'Outdoor',
     'Transport', 'Healthcare', 'Services', 'Organization',
     'Education', 'Religion', 'Home Improvement']
    )


def prepare_features():
    input_dict = {
        'borough': borough,
        'property_type': property_type,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'minimum_nights': minimum_nights,
        'distance_to_station': distance_to_station,
        'amenity_1': amenity_1,
        'amenity_2': amenity_2,
        'amenity_3': amenity_3
    }
    return pd.DataFrame([input_dict])


if st.button("Predict Price"):
    features_df = prepare_features()

    try:
        prediction_log = model.predict(features_df).iloc[0]
        prediction = np.expm1(prediction_log)
        st.success(f"Recommended Price: "
                   f"**£{prediction:.2f}** per night")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
