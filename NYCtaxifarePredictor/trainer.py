import pandas as pd
import numpy as np
from NYCtaxifarePredictor.data import get_data, clean_train
from NYCtaxifarePredictor.transformer import TimeFeatures, DistanceFeatures, DirectionFeatures, AirportFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBRegressor
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from termcolor import colored
import time

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = '[DE][Berlin][Guli]NYCtaxifarePredictor'

class Trainer():
    X = ['pickup_datetime',
         'pickup_longitude', 'pickup_latitude',
         'dropoff_longitude', 'dropoff_latitude']
    y = 'fare_amount'
    def __init__(self, df, **kwargs):
        self.kwargs = kwargs
        self.mlflow = self.kwargs.get('mlflow', False)
        self.estimator = self.kwargs.get('estimator', 'xgboost')
        self.estimator_params = self.kwargs.get('estimator_params',
                                                dict(learning_rate=[0.1],
                                                    n_estimators=[100],
                                                    max_depth=[5],
                                                    min_child_weight=[2]))
        self.run_name = self.kwargs.get('run_name', 'unknown')
        self.nrows = df.shape[0]
        print(colored('---------------------- loading data  ----------------------', 'green'))
        self.pipeline = None
        self.test = None
        self.train_time = None
        self.split = self.kwargs.get("split", True)
        print(colored(f'---------------------- split = {self.split} ----------------------', 'red'))
        if self.split:
            self.split_params = self.kwargs.get('split_params',
                                            dict(test_size=0.1,
                                                random_state=5))
            print(self.split_params)
            print(colored('---------------------- splitting train/val  ----------------------', 'blue'))
            self.train, self.test = train_test_split(df, **self.split_params)
            self.test = self.test.dropna()
        else:
            self.train = df
        del df
        print(colored('---------------------- cleaning data  ----------------------', 'blue'))
        self.train = clean_train(self.train)

    def get_estimator(self):
        """return estimator"""
        print(colored('---------------------- getting estimator  ----------------------', 'blue'))
        if self.estimator == 'xgboost':
            model = XGBRegressor(objective='reg:squarederror',
                               random_state=5,
                               verbosity=1,
                               booster='gbtree')
        print(colored(f'---------------------- {model.__class__.__name__} ----------------------', 'red'))
        return model

    def get_pipeline(self):
        """define pipeline here, define it as a class attribute"""
        print(colored('---------------------- getting pipeline  ----------------------', 'blue'))
        location_cols = ['pickup_longitude', 'pickup_latitude',
                     'dropoff_longitude', 'dropoff_latitude']
        distance = Pipeline([
            ('distance', DistanceFeatures()),
            ('scaler', StandardScaler())
        ])
        preprocessor = ColumnTransformer([
            ('time', TimeFeatures(), ['pickup_datetime']),
            ('distance', distance, location_cols),
            ('direction', DirectionFeatures(), location_cols),
            ('airport', AirportFeatures(), location_cols)
        ])
        self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', self.get_estimator())
                ])

    def train_model(self):
        self.get_pipeline()
        grid_params = \
                    {'model__'+key: val for key, val in self.estimator_params.items()}
        clf = GridSearchCV(self.pipeline, param_grid=grid_params, cv=5,
                            scoring='neg_root_mean_squared_error', verbose=1)
        print(colored('---------------------- start training  ----------------------', 'green'))
        start = time.time()
        self.pipeline = clf.fit(self.train[self.X], self.train[self.y])
        end = time.time()
        self.train_time = int(end-start)
        print('Grid Scores:')
        means = self.pipeline.cv_results_['mean_test_score']
        stds = self.pipeline.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, self.pipeline.cv_results_['params']):
            print(f'rmse = {-round(mean,3)} (+/-{round(std,3)}) for {params}')
        print(colored('---------------------- best_params ----------------------', 'red'))
        print(self.pipeline.best_params_)
        print('train time:', self.train_time)

    def get_rmse(self, df):
        if self.pipeline is None:
            print(colored("The model hasn't been trained yet", 'red'))
        if self.test is None:
            print(colored("Test data not available. Please consider set split parameter to True.", 'red'))
        y_pred = self.pipeline.predict(df[self.X])
        y_true = df[self.y]
        return round(np.sqrt(np.mean((y_true-y_pred)**2)), 3)

    def evaluate(self):
        print(colored('---------------------- evaluating  ----------------------', 'green'))
        test_rmse = self.get_rmse(self.test)
        train_rmse = self.get_rmse(self.train)
        print(colored(f'---------------------- train_RMSE: {train_rmse}  ----------------------', 'red'))
        print(colored(f'----------------------  test_RMSE: {test_rmse}  ----------------------', 'red'))
        if self.mlflow:
            print(colored('---------------------- logging params and metrics  ----------------------', 'green'))
            # logging params and metrics of each search in grid_search
            means = self.pipeline.cv_results_['mean_test_score']
            stds = self.pipeline.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, self.pipeline.cv_results_['params']):
                with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.run_name+'_grid_search'):
                    for key, val in params.items():
                        mlflow.log_param(key[7:], val)
                    mlflow.log_metric('cv_RMSE', -mean)
            # logging params and metrics of trainer
            with mlflow.start_run(experiment_id=self.mlflow_experiment_id, run_name=self.run_name):
                mlflow.log_param('nrows', self.nrows)
                mlflow.log_param('split', self.split)
                if self.split:
                    mlflow.log_param('split_params', self.split_params)
                mlflow.log_param('estimator', self.estimator)
                mlflow.log_param('grid_params', self.estimator_params)
                for key, val in self.pipeline.best_params_.items():
                    mlflow.log_param('best' + key[5:], val)
                mlflow.log_metric('train_RMSE', train_rmse)
                mlflow.log_metric('test_RMSE', test_rmse)
                mlflow.log_metric('train_time', self.train_time)

    def predict(self, test):
        if self.pipeline is None:
            print(colored("The model hasn't been trained yet", 'red'))
        else:
            test['fare_amount'] = self.pipeline.predict(test[self.X])
            test.set_index('key', inplace=True)
            test[['fare_amount']].to_csv('../result/taxifare_prediction_result.csv')

    def save_model(self):
        if self.pipeline is None:
            raise ("No model hasn't been trained")
        joblib.dump(self.pipeline, self.estimator+'.joblib')
        print(colored("Model saved locally", 'green'))

    ############################# MLflow part #################################
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        experiment_name = self.kwargs.get('experiment_name', EXPERIMENT_NAME)
        try:
            return self.mlflow_client.create_experiment(experiment_name)
        except:
            return self.mlflow_client.get_experiment_by_name(experiment_name).experiment_id


if __name__=='__main__':
    df = get_data(n=1000)
    xgb = Trainer(df)
    xgb.train_model()
    xgb.evaluate()
