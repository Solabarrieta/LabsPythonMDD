import numpy as np
import pandas as pd
import statistics as st
import math
import csv
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def main():
    path_dataset = "mtcars.csv" # Escoged bien la ruta!!
    mtcars = pd.read_csv(path_dataset) # Leemos el csv
    # Discretizamos la variable clase para convertirlo en un problema de clasificacion
    ix_consumo_alto = mtcars.mpg >= 21
    mtcars.mpg[ix_consumo_alto] = 1
    mtcars.mpg[~ix_consumo_alto] = 0
    print("Este es el dataset sin normalizar")
    print(mtcars)
    print("\n\n")
    # Ahora normalizamos los datos
    mtcars_normalizado = mtcars.loc[:, mtcars.columns != 'mpg'].apply(normalize, axis=1)
    # Añadimos la clase a nuestro dataset normalizado
    mtcars_normalizado['mpg'] = mtcars['mpg']
    print("Este es el dataset normalizado")
    print(mtcars_normalizado)
    print("\n\n")
    # Hacemos un split en train y test con un porcentaje del 0.75 Train
    train, test = splitTrainTest(mtcars_normalizado, 0.75)
    print("Datos de train: ")
    print(train)
    print("\n\n")
    print("Datos de test: ")
    print(test)
    print("\n\n")


    # Separamos las labels del Test. Es como si no nos las dieran!!
    true_labels = test['mpg']
    test = test.loc[:, test.columns != 'mpg']

    # Predecimos el conjunto de test
    K=5
    predicted_labels = []
    for i in range(len(test)):
        row = test.iloc[i,:]
        predicted_labels.append(knn(row, train, K))

    # Mostramos por pantalla el Accuracy por ejemplo
    print("Accuracy conseguido:")
    print(accuracy(true_labels, predicted_labels))

    # Algun grafico? Libreria matplotlib.pyplot
    return(0)

# FUNCIONES de preprocesado
def normalize(x):
    return((x-min(x)) / (max(x) - min(x)))

def standardize(x):
    return((x-st.mean(x))/st.variance(x))

# FUNCIONES de evaluacion
def splitTrainTest(data, percentajeTrain):
    """
    Takes a pandas dataframe and a percentaje (0-1)
    Returns both train and test sets
    """
    msk = np.random.rand(len(data)) < percentajeTrain
    train = data[msk]
    test = data[~msk]
    return train, test

def kFoldCV(data, K):
    """
    Takes a pandas dataframe and the number of folds of the CV
    YOU CAN USE THE sklearn KFold function here
    How to: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    return()

# FUNCION modelo prediccion
def knn(newx, data, K):
    """
    Receives two pandas dataframes. Newx consists on a single row df.
    Returns the prediction for newx
    """
    distances = []

    for i in range(len(data)):
        distance = euclideanDistance2points(newx.values, data.iloc[i, :-1].values)
        distances.append((distance, data.iloc[i, -1]))

    distances = sorted(distances, key=lambda x: x[0][0])

    nearest_neighbors = distances[:K]

    clases = {}

    for neighbor in nearest_neighbors:

        if neighbor[1] not in clases:
            clases[neighbor[1]] = 1
        else:
            clases[neighbor[1]] += 1
    
    return max(clases, key=clases.get)


def euclideanDistance2points(x,y):
    """
    Takes 2 matrix - Not pandas dataframe!
    """
    restaArrays = x-y
    alcuadrado = np.power(restaArrays,2)
    sumaCuadrada = np.sum(alcuadrado)
    distancia = np.sqrt(alcuadrado)
    return distancia

# FUNCION accuracy
def accuracy(true, pred):
    if len(true) != len(pred):
        return 0
    
    correct = 0

    for i in range(len(true)):
        if true[i] == pred[i]:
            correct+=1
    accuracy = correct/len(true)

    return accuracy

if __name__ == '__main__':
    np.random.seed(4)
    main()
