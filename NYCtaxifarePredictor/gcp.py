from google.cloud import storage
import joblib
from termcolor import colored
import os

BUCKET_NAME = 'nyc_taxifare_predictor'
MODEL_NAME = 'xgboost'
VERSION_NAME = 'RunNo1'

def load_model(model_name=MODEL_NAME, version_name=VERSION_NAME):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'models/{model_name}/{version_name}/model.joblib')
    blob.download_to_filename('model.joblib')
    print(colored(f'------------ downloaded the trained model from storage ------------', 'blue'))
    model = joblib.load('model.joblib')
    os.remove('model.joblib')
    return model
