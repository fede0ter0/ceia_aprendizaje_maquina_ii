"""
train.py

DESCRIPTION:
This is a script to train a linear regression model with preprocessed data 
obtained from feature_engineering, provided through a specific path. This 
location of the preprocessed data, along with the location where the trained 
model should be saved, are the parameters passed to the file.

AUTHORS:
    Grupo AdMII: 
        - Federico Otero - fede.e.otero@gmail.com, 
        - Rodrigo Carranza - rodrigocarranza81@gmail.com, 
        - Agustin Menara - menaragustin@gmail.com
        
DATE: 07/10/2023
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import utils as u
import pickle

logger = u.make_logger(__name__)


class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        :description: This method retrieves pre-processed data from a .csv file located on the disk.
        :parameters: None
        :return: A data frame with pre-processed data
        :rtype: Pandas DataFrame        
        """

        data = pd.read_csv(self.input_path)
        # logger.info(f'lets see the file: \n {data.head()} ')
        return data

    def model_training(self, df: pd.DataFrame):
        """
        :description: This method aims to train a linear regression model using preprocessed data.
        :parameters: 
        df: Pandas DataFrame : Pre processed training data
        :return model: A trained regression model 
        :rtype: An object of class LinearRegression
        """
        logger.info('Training model...')
        model = LinearRegression()

        train_data = df[df['Set'] == 'train']
        test_data = df[df['Set'] == 'test']

        x_train = train_data.drop(['Item_Outlet_Sales', 'Set'], axis=1).copy()
        y_train = train_data[['Item_Outlet_Sales']].copy()
        x_val = test_data.drop(['Item_Outlet_Sales', 'Set'], axis=1).copy()

        # Model training
        model.fit(x_train, y_train)

        # A prediction from the fitted model for the validation set is made to get an idea of its performance.
        pred = model.predict(x_val)

        # Calculation of Mean Squared Errors and Coefficient of Determination (R^2)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        logger.info('Model metrics:')
        logger.info(
            'TRAINING: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

        # mse_val = metrics.mean_squared_error(y_val, pred)
        # R2_val = model.score(x_val, y_val)
        # logger.info('VALIDATION: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

        # Model constant
        logger.info(f'Intercept: {model.intercept_}')

        # Model Coefficients
        # coef = pd.DataFrame(x_train.columns, columns=['features'])
        # coef['Estimated Coeficients'] = model.coef_.reshape[-1,1]
        logger.info(f'Coefficients: {model.coef_[0]}')
        # coef.sort_values(by='Estimated Coeficients').set_index('features').plot(kind='bar', title='Variable Importance', figsize=(12, 6))
        # plt.show()
        return model

    def model_dump(self, model_trained) -> None:
        """
        :description: This method saves a pre-trained regression model to disk.
        :parameters:
        model_trained: pre trained model
        :return: None
        """

        logger.info('Serializing model...')
        with open(self.model_path, 'wb') as model_file:
            pickle.dump(model_trained, model_file)

    def run(self):

        logger.info('Reading preprocessed data to be trained...')
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)
        logger.info('Finish serializing model OK.')
