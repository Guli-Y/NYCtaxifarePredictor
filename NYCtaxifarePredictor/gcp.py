from google.cloud import storage
from google.oauth2 import service_account
import json
import joblib
from termcolor import colored
import os

BUCKET_NAME = 'nyc_taxifare_predictor'
MODEL_NAME = 'xgboost'
VERSION_NAME = 'RunNo6'

def get_credentials():
    key = 'GOOGLE_APPLICATION_CREDENTIALS'
    if key in os.environ:
        cred_file = os.environ.get(key)
    else:
        print('Google application credentials not found')
    if '.json' in cred_file:
        cred_file = open(cred_file).read()
    cred_json = json.loads(cred_file)
    cred_gcp = service_account.Credentials.from_service_account_info(cred_json)
    return cred_gcp

def load_model(model_name=MODEL_NAME, version_name=VERSION_NAME):
    client = storage.Client(credentials=get_credentials(), project='wagon-project-guli')
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'models/{model_name}/{version_name}/model.joblib')
    blob.download_to_filename('model.joblib')
    print(colored(f'------------ downloaded the trained model from storage ------------', 'blue'))
    model = joblib.load('model.joblib')
    os.remove('model.joblib')
    return model

if __name__ == '__main__':
    get_credentials()
