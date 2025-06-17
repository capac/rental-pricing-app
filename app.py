import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load('model.pkl')


model = load_model()

st.title('Short-Term Rental Price Estimator')

st.write(
    'Use this tool to estimate the short-term rental nightly '
    'price of your property based on its characteristics and '
    'location in London, UK.'
    )

st.write(
    '**Short-term rentals are those where the nights stay is from '
    '1 to 999 days. The data set used for modeling is from '
    'December 2024.**'
)

property_type = st.selectbox(
    'Property type',
    ['Entire rental unit', 'Entire condo', 'Private room in home',
     'Private room in rental unit', 'Entire townhouse',
     'Private room in townhouse', 'Private room in condo',
     'Private room in bed and breakfast', 'Entire home',
     'Entire guest suite', 'Private room in serviced apartment',
     'Entire loft', 'Private room in guesthouse', 'Entire serviced apartment',
     'Private room in loft', 'Private room in guest suite',
     'Entire guesthouse', 'Room in hotel']
    )

room_type = st.selectbox(
    'Room type',
    ['Entire home/apt', 'Private room'])

accommodates = st.number_input(
    'Number of people the property can accommodate', min_value=0,
    max_value=20, value=1, step=1
    )

bedrooms = st.number_input(
    'Number of bedrooms', min_value=0, max_value=20, value=1, step=1
    )

bathrooms = st.number_input(
    'Number of bathrooms', min_value=0, max_value=20, value=1, step=1
    )

borough = st.selectbox(
    'London borough',
    ['Barking and Dagenham', 'Barnet', 'Bexley', 'Brent',
     'Bromley', 'Camden', 'City of London', 'Croydon', 'Ealing',
     'Enfield', 'Greenwich', 'Hackney', 'Hammersmith and Fulham',
     'Haringey', 'Harrow', 'Havering', 'Hillingdon', 'Hounslow',
     'Islington', 'Kensington and Chelsea', 'Kingston upon Thames',
     'Lambeth', 'Lewisham', 'Merton', 'Newham', 'Redbridge',
     'Richmond upon Thames', 'Southwark', 'Tower Hamlets',
     'Waltham Forest', 'Wandsworth', 'Westminster']
     )

availability_365 = st.slider(
    'Current yearly availabiliy',
    min_value=0, max_value=365, value=0, step=1
    )

days_from_last_review = st.slider(
    'Days from last review (maximum 6 months)',
    min_value=0, max_value=182, value=0, step=1
    )

distance_to_station = st.slider(
    'Distance to nearest station (km)',
    min_value=0.0, max_value=10.0, value=0.0, step=0.1
    )

first_amenity = st.selectbox(
    'First nearby amenity',
    ['Dining and Drinking', 'Community and Government', 'None',
     'Health and Medicine', 'Arts and Entertainment', 'Retail',
     'Business and Professional Services', 'Landmarks and Outdoors',
     'Sports and Recreation', 'Travel and Transportation', 'Event']
    )

second_amenity = st.selectbox(
    'Second nearby amenity',
    ['Dining and Drinking', 'Community and Government', 'None',
     'Health and Medicine', 'Arts and Entertainment', 'Retail',
     'Business and Professional Services', 'Landmarks and Outdoors',
     'Sports and Recreation', 'Travel and Transportation', 'Event']
    )

third_amenity = st.selectbox(
    'Third nearby amenity',
    ['Dining and Drinking', 'Community and Government', 'None',
     'Health and Medicine', 'Arts and Entertainment', 'Retail',
     'Business and Professional Services', 'Landmarks and Outdoors',
     'Sports and Recreation', 'Travel and Transportation', 'Event']
    )

crime_rate = {
    'Barking and Dagenham': 115.58, 'Barnet': 89.86, 'Bexley': 86.95,
    'Brent': 117.59, 'Bromley': 88.26, 'Camden': 108.07,
    'City of London': 28.93, 'Croydon': 111.91, 'Ealing ': 103.37,
    'Enfield': 104.58, 'Greenwich': 121.4, 'Hackney': 140.13,
    'Hammersmith and Fulham': 103.14, 'Haringey': 133.11,
    'Harrow': 79.15, 'Havering': 99.07, 'Hillingdon': 95.3,
    'Hounslow': 102.7, 'Islington': 114.9, 'Kensington and Chelsea': 118.02,
    'Kingston upon Thames': 75.43, 'Lambeth': 137.98, 'Lewisham': 134.35,
    'Merton': 82.49, 'Newham': 142.35, 'Redbridge': 101.74,
    'Richmond upon Thames': 71.78, 'Southwark': 116.55, 'Sutton': 79.67,
    'Tower Hamlets': 98.6, 'Waltham Forest': 112.25, 'Wandsworth': 106.6,
    'Westminster': 132.94}


def prepare_features():
    input_dict = {
        'property_type': property_type,
        'room_type': room_type,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'accommodates': accommodates,
        'borough': borough,
        'crime_rate': crime_rate[borough],
        'distance_to_nearest_tube_station': distance_to_station,
        'availability_365': availability_365,
        'days_from_last_review': days_from_last_review,
        'first_amenity': first_amenity,
        'second_amenity': second_amenity,
        'third_amenity': third_amenity
    }
    return pd.DataFrame([input_dict])


if st.button("Predict Price"):
    features_df = prepare_features()

    try:
        prediction_log = model.predict(features_df)[0]
        prediction = np.expm1(prediction_log)
        st.success(f"Recommended Price: "
                   f"**£{prediction:.2f}** per night")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
