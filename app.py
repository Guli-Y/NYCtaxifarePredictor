import streamlit as st
from datetime import datetime
import pandas as pd
import joblib
import herepy
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
HERE_API_KEY = os.getenv('HERE_API_KEY')

X = ['pickup_datetime',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude']

def geocoder(address, key= HERE_API_KEY):
    geoapi = herepy.GeocoderApi(api_key=key)
    result = geoapi.free_form(address).as_dict()
    coords = result["items"][0]["position"]
    coords = {k.lower().replace('lng', 'lon'):v for k, v in coords.items()}
    return coords

def format_input(pickup, dropoff):
    pickup_datetime = datetime.utcnow()
    formated_input = {'pickup_datetime': str(pickup_datetime)+' UTC',
                        'pickup_latitude': float(pickup['lat']),
                        'pickup_longitude': float(pickup['lon']),
                        'dropoff_latitude': float(dropoff['lat']),
                        'dropoff_longitude': float(dropoff['lon'])}
    return formated_input

def main():
    st.set_page_config(page_title="nyc-taxifare-predictor",
                    page_icon=":oncoming_taxi:",
                    layout="centered")
    st.markdown('https://github.com/Guli-Y/NYCtaxifarePredictor')
    pipe = joblib.load('model.joblib')
    print('------------ loaded model ---------------')
    st.header('NYC Taxi Fare Predictor :taxi:')
    st.write('Please type in pickup and dropoff locations to get predicted taxi fare amount!')
    # input
    pickup_address = st.text_input('pickup address', '45 Rockefeller Plaza, New York, NY 10111')
    dropoff_address = st.text_input('dropoff address', '334 Furman St, Brooklyn, NY 11201')
    # get coords
    pickup_coords = geocoder(pickup_address)
    dropoff_coords = geocoder(dropoff_address)
    # input dictionary
    df = pd.DataFrame([format_input(pickup_coords, dropoff_coords)])
    df = df[X]
    result = pipe.predict(df)
    fare = round(float(result[0]), 3)
    st.write(':oncoming_taxi: *Fare Amount*', fare, ':heavy_dollar_sign:')
    locations = pd.DataFrame([pickup_coords, dropoff_coords])
    st.map(data=locations, zoom=11)

if __name__ == '__main__':
    main()
