"""
train_pipeline.py

DESCRIPTION:
Script that runs the train pipeline. It consists in two steps: feature_engineering and train. The first one
prepares the data and the second one runs and pickels a linear regression model.

AUTHORS:
    Grupo AdMII: 
        - Federico Otero - fede.e.otero@gmail.com, 
        - Rodrigo Carranza - rodrigocarranza81@gmail.com, 
        - Agustin Menara - menaragustin@gmail.com

DATE: 10-13-2023
"""

# Imports
import argparse
import steps.feature_engineering as feat
import steps.train as train
import utils as u

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Script that runs the training of the model")

    parser.add_argument(
        "-itr",
        "--input_train",
        type=str,
        help="input train path",
        required=True)

    parser.add_argument(
        "-ite",
        "--input_test",
        type=str,
        help="input test path",
        required=False)
    

    parser.add_argument(
        "-o",
        "--output",
        default=0,
        type=str,
        help="output path",
        required=True)
    
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="model path",
        required=True)

    args = parser.parse_args()
    input_train, input_test, output, model = args.input_train, args.input_test, args.output, args.model
    logger = u.make_logger(__name__)    
    feat.FeatureEngineeringPipeline(input_train_path = input_train,
                               input_test_path = input_test,
                               output_path = output).run()
    
    train.ModelTrainingPipeline(output,model).run()