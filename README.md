# Aprendizaje de Máquina II

## CEIA - FIUBA

### Trabajo final

#### Grupo Cohorte 11
- Federico Otero - fede.e.otero@gmail.com
- Rodrigo Carranza - rodrigocarranza81@gmail.com
- Agustin Menara - menaragustin@gmail.com

#### Explicación general de la solución 

Seguimos la recomendación del enunciado del trabajo práctico, y respetamos todos los encabezados de las funciones propuestas para los métodos de los pipelines, de modo que tenemos principalmente dos pipelines:

1. **train_pipeline.py**: Entrena el modelo, y se divide a su vez en dos pasos:
  * **feature_engineering**: Hace una preparación de los datos.
  * **train.py**: Utiliza los datos preparados previamente para entrenar un modelo y lo almacena seralizado en formato _pkl_
2. **inference_pipeline.py**: Permite calcular predicciones para nuevos datos, y se también divide en dos pasos:
  * **feature_engineering.py**: Limpiamos los datos de entrada antes de aplicar el modelo.
  * **predict.py**: Calcula predicciones a partir de un dataset.

Tratamos de seguir las buenas prácticas de diseño y desarrollo. Aplicamos el `autopep8` para estandarizar los archivos `.py`, y elegimos no aplicar Optuna ya que la regresión lineal contiene demasiados pocos hiperpárametros, que pueden ajustarse fácilmente de manera manual. Utilizamos la libería `logging` para ir imprimiendo mensajes de salida de control de la ejecución. Además, utilizamos scripts de python que pueden recibir parámetros con `argparse`. 

#### Pasos previos a correr los pipelines

Necesitamos crear y activar un entorno virtual desde nuestra consola. Para eso, corremos el comando

`python -m venv path/to/venv`

Luego, activamos el entorno virtual

`source path/to/venv/bin/activate`

Y cargamos las librerías

`pip install -r requirements.txt`

#### Ejecución de los pipelines

Como veremos, ambos pipelines utilizar un logger para ir haciendo un seguimiento controloado de los pasos en cada proceso de ejecución. 

Para ejecutar el entrenamiento del modelo, veamos un ejemplo corrido desde la consola:

`python train_pipeline.py -itr ../data/raw/Train_BigMart.csv -ite ../data/raw/Test_BigMart.csv -o ../data/pre_processed/Preprocessed_BigMart.csv -m ../model/pickle_model.pkl`

Tenemos que pasar como parámetros:
```
-itr: Es el input path del dataset de training.
-ite: Es el input path del dataset de testing.
-o: Es el ouput path a la carpeta pre_preocessed.
-m: Es el path del modelo serializado en formato .pkl.
```

Para ejecutar una predicción de un dataset, veamos un ejemplo corrido desde la consola:

`python3 inference_pipeline.py -itr ../data/raw/Train_BigMart.csv -ite ../data/raw/Test_BigMart.csv -o ../data/pre_processed/Preprocessed_BigMart.csv -m ../model/pickle_model.pkl -p ../data/predictions/Predicted_BigMart.csv`

Tenemos que pasar como parámetros:
```
-itr: Es el input path del dataset de training.
-ite: Es el input path del dataset de testing.
-o: Es el ouput path a la carpeta pre_preocessed.
-m: Es el path del modelo serializado en formato .pkl.
-p: Es el path del dataset de salida con las predicciones ejecutadas.
```

