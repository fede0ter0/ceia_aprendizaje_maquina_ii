"""
predict.py

DESCRIPTION:
Scrip to make predictions from csv file.
It is required to pass a path to the file with the samples to be predicted,
another path where the predictions and the path to the serialized model to use.

AUTHORS:
    Grupo AdMII: 
        - Federico Otero - fede.e.otero@gmail.com, 
        - Rodrigo Carranza - rodrigocarranza81@gmail.com, 
        - Agustin Menara - menaragustin@gmail.com

DATE: 07/10/2023
"""

# Imports
import pickle
import pandas as pd
import steps.feature_engineering as feat
import utils as u

logger = u.make_logger(__name__)

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path                
                
    def load_data(self) -> pd.DataFrame:
        """
        :description: Reads the data to be predicted from the input_path.
        The feature engineering pipeline is applied before returning data.
        :parameters: None      
        :return data: dataframe with the data to predict.
        :rtype: pd.DataFrame
        """

        data = pd.read_csv(self.input_path)
        self.data = data
        logger.info('Lets see the dataframe to make predictions:')
        logger.info(data.info())

    def load_model(self) -> None:
        """
        :description: Method to load the model as an attribute of the class.
        :parameters: None      
        :return None
        """
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)
        logger.info('The model was loaded OK.')

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :description: Consults the model and make predictions.
        :parameters: 
        data: dataframe with the data to predict.  
        :return new_data:  dataframe with the id column of the samples to predict and its prediction
        :rtype: pd.DataFrame
        """
        test_data = data[data['Set']=='test']
        x_test = test_data.drop(['Item_Outlet_Sales','Set'], axis=1).copy()
        y_test = self.model.predict(x_test)
        logger.info(f'Lets see the predictions:')
        logger.info(y_test)
        predicted_data = pd.DataFrame(data = y_test,columns=['Item_Outlet_Sales'])
        logger.info(f'Lets see predicted data:')
        logger.info(f'{predicted_data.head()}')
        return predicted_data

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        :description: Writes predictions.
        :parameters: 
        predicted_datata: dataframe with id and predictions.  
        :return None.
        """
        test_data = self.data[self.data['Set']=='test']
        x_test = test_data.drop(['Item_Outlet_Sales','Set'], axis=1).copy()
        test_data = x_test.copy()
        test_data['Item_Outlet_Sales'] = predicted_data['Item_Outlet_Sales'].tolist()
        logger.info('We are going to write this dataframe:')
        logger.info(f'{test_data.info()}')
        test_data.to_csv(self.output_path)

    def run(self):
        logger.info('Starting to run predict.py...')
        logger.info('Reading Data...')
        self.load_data()
        logger.info('Loading model...')
        self.load_model()
        logger.info('Making predictions...')
        df = self.make_predictions(self.data)
        logger.info('Write predictions...')
        self.write_predictions(df)
        logger.info('Finished OK.')