from NYCtaxifarePredictor.data import get_test_data
from NYCtaxifarePredictor.gcp import load_model
from termcolor import colored
import os

VERSION_NAME = 'RunNo8'

def predict():
    test = get_test_data()
    test.set_index('key', inplace=True)
    model = load_model(version_name=VERSION_NAME)
    test['fare_amount'] = model.predict(test)
    result = test[['fare_amount']]
    result.to_csv('prediction.csv')
    print(colored('------------------ prediction saved as csv file -----------------', 'green'))

def kaggle_upload():
    command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f "prediction.csv" -m "{VERSION_NAME}_trained with whole dataset"'
    os.system(command)

if __name__ == '__main__':
    predict()
    kaggle_upload()
