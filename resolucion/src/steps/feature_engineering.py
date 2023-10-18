"""
feature_engineering.py

DESCRIPTION:
Script that prepares the original dataset to be used in a predictive model

AUTHORS:
    Grupo AdMII: 
        - Federico Otero - fede.e.otero@gmail.com, 
        - Rodrigo Carranza - rodrigocarranza81@gmail.com, 
        - Agustin Menara - menaragustin@gmail.com

DATE: 10-08-2023
"""

# Imports
import pandas as pd
import utils as u

logger = u.make_logger(__name__)


class FeatureEngineeringPipeline(object):

    def __init__(self, input_train_path, input_test_path, output_path):
        self.input_train_path = input_train_path
        self.input_test_path = input_test_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        :description: Reads the original dataset and returns a pandas DataFrame
        :parameters: None      
        :return data: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        # It checks if the original dataset is already split into train and test
        if self.input_test_path:
            data_train = pd.read_csv(self.input_train_path)
            data_test = pd.read_csv(self.input_test_path)
            data_train['Set'] = 'train'
            data_test['Set'] = 'test'
            data = pd.concat([data_train, data_test],
                             ignore_index=True, sort=False)
        else:
            data = pd.read_csv(self.input_train_path)

        logger.info(f'Lets see a sample of the data: \n {data.head()}')
        return data

    def replace_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        :description: Replace null values in the features of the DataFrame
        :parameters: 
        df: Pandas DataFrame 
        :return df:
        :rtype: pd.DataFrame
        """

        logger.info('Replacing null values in the dataset...')
        products_ids = list(df[df['Item_Weight'].isnull()]
                            ['Item_Identifier'].unique())
        for _id in products_ids:
            _mode = (df[df['Item_Identifier'] == _id]
                     [['Item_Weight']]).mode().iloc[0, 0]
            df.loc[df['Item_Identifier'] == _id, 'Item_Weight'] = _mode

        outlets = list(df[df['Outlet_Size'].isnull()]
                       ['Outlet_Identifier'].unique())
        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] = 'Small'

        return df

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        :description: Apply transformations to some features
        :parameters: 
        df: Pandas DataFrame 
        :return df:
        :rtype: pd.DataFrame
        """

        logger.info('Transforming Outlet_Type and Item_MRP...')

        df['Outlet_Type'] = df['Outlet_Type'].replace(
            {'Supermarket Type1': 'Supermarket_Type1', 'Supermarket Type2': 'Supermarket_Type2',
             'Supermarket Type3': 'Supermarket_Type3', 'Grocery Store': 'Grocery_Store'}
        )

        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels=[1, 2, 3, 4])
        df['Item_MRP'] = df['Item_MRP'].astype(int)

        df['Years_Since_Establishment'] = 2023 - \
            df['Outlet_Establishment_Year']
        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        :description: Encode categorial variables of the DataFrame
        :parameters: 
        df: Pandas DataFrame 
        :return df:
        :rtype: pd.DataFrame
        """

        logger.info('Encoding Outlet_Size and Outlet_Location_Type...')
        df['Outlet_Size'] = df['Outlet_Size'].replace(
            {'High': 2, 'Medium': 1, 'Small': 0})
        df['Outlet_Location_Type'] = df['Outlet_Location_Type'].replace(
            {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})
        df = pd.get_dummies(df, columns=['Outlet_Type'], dtype=int)
        return df

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        :description: Apply transformations to the original dataset in order to have appropiate data to feed the model
        :parameters: 
        df: Pandas DataFrame 
        :return df_transformed:
        :rtype: pd.DataFrame
        """

        # Replace null values
        df = self.replace_null_values(df)

        # Features transformations
        df = self.transform_features(df)

        # Encoding categorial features
        df = self.encode_features(df)

        # Delete columns
        df_transformed = df.drop(columns=['Item_Type', 'Item_Fat_Content', 'Item_Identifier',
                                          'Outlet_Identifier', 'Outlet_Establishment_Year']).copy()

        return df_transformed

    def write_prepared_data(self, df_transformed: pd.DataFrame) -> None:
        """
        :description: Write the data recently transformed to the pre_processed folder in csv format
        :parameters: 
        df_transformed: Pandas DataFrame
        """

        # Reorder columns
        df_to_write = df_transformed[['Item_Weight', 'Item_Visibility', 'Item_MRP',
                                      'Years_Since_Establishment', 'Outlet_Size',
                                      'Outlet_Location_Type', 'Outlet_Type_Grocery_Store', 'Outlet_Type_Supermarket_Type1',
                                      'Outlet_Type_Supermarket_Type2', 'Outlet_Type_Supermarket_Type3', 'Item_Outlet_Sales',
                                      'Set']]

        logger.info('Output dataset sample:')
        logger.info(df_to_write.head())
        logger.info('Output dataset information:')
        logger.info(df_to_write.info())
        df_to_write.to_csv(self.output_path, sep=',', index=False)
        logger.info(
            f'Data was preprocessed OK and written in {self.output_path}')

    def run(self):
        """
        :description: Runs the three steps defined above: read, transform and write.
        """
        logger.info('Reading data...')
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)
