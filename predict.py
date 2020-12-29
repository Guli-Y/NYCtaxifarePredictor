from NYCtaxifarePredictor.data import get_test_data
from termcolor import colored
import os

BUCKET_NAME = 'nyc_taxifare_predictor'
MODEL_NAME = 'xgboost'
VERSION_NAME = 'test'
KAGGLE_MESSAGE = 'test'

def load_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = client.blob(f'models/{MODEL_NAME}/{VERSION_NAME}/model.joblib')
    blob.download_to_filename('model.joblib')
    print(colored(f'------------ downloaded the trained model from storage ------------', 'blue'))
    model = joblib.load('model.joblib')
    os.remove('model.joblib')
    return model

def predict():
    test = get_test_data()
    model = load_model()
    test['fare_amount'] = model.predict(test)
    test.set_index('key', inplace=True)
    result = test[['fare_amount']]
    result.to_csv('prediction.csv')
    print(colored('------------------ prediction saved as csv file -----------------', 'green'))

def kaggle_upload():
    command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f "prediction.csv" -m "{KAGGLE_MESSAGE}"'
    os.system(command)

if __name__ = '__main__':
    predict()
