"""
predict.py

DESCRIPCIÓN:

Scrip para realizar predicciones a partir de un csv.
Se requiere pasar un path al archivo con las muestras a predecir,
otro path donde se guardan las predicciones y el path al modelo
serializado a utilizar.

AUTOR/ES:
    Grupo AdMII: 
        - Federico Otero - fede.e.otero@gmail.com, 
        - Rodrigo Carranza - rodrigocarranza81@gmail.com, 
        - Agustin Menara - menaragustin@gmail.com

FECHA: 07/10/2023
"""

# Imports
import pickle
import pandas as pd
from feature_engineering import FeatureEngineeringPipeline


class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self) -> pd.DataFrame:
        """
        Metodo para leer la data a predecir a partir del input_path.
        Se aplica feature_engineering antes de devolver la data

        Parametros:
            - None

        Return:
            - data (pd.DataFrame): dataframe con los datos a predecir
        """

        extension = '.' + self.input_path.split('.')[-1]
        out_fe_path = self.input_path.replace(extension,f'_fe{extension}')

        fe_o = FeatureEngineeringPipeline(input_path = self.input_path,
                                          output_path = out_fe_path)
        fe_o.run()

        data = pd.read_csv(out_fe_path)

        return data

    def load_model(self) -> None:
        """
        Metodo para cargar el modelo como un atributo de la clase.

        Parametros:
            - None

        Return:
            - None
        """    
        
        # Load from file
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file) 
        
        return None


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Metodo para consultar al modelo y realizar las predicciones.

        Parametros:
            - data (pd.DataFrame): dataframe con las muestras que se quiere predecir

        Return:
            - new_data (pd.DataFrame): dataframe con la columna id de las muestras
                                       a predecir y su prediccion
        """
   
        predictions = self.model.predict(data)
        new_data = pd.DataFrame(predictions,columns=['pred_Sales'])
        

        return new_data


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Metodo para guardar las predicciones calculadas.

        Parametros:
            - predicted_data (pd.DataFrame): dataframe con id y prediccion calculada

        Return:
            - None
        """

        predicted_data.to_csv(self.output_path)

        return None


    def run(self):
        print('Getting data')
        data = self.load_data()
        print('Loading model')
        self.load_model()
        print('Make predictions')
        df_preds = self.make_predictions(data)
        print('Write preds')
        self.write_predictions(df_preds)
        print('DONE')


if __name__ == "__main__":
        
    pipeline = MakePredictionPipeline(input_path = '../data/data_test_sample.csv',
                                      output_path = '../data/Test_BigMart_predicciones.csv',
                                      model_path = '../model/pickle_model.pkl')
    pipeline.run()  