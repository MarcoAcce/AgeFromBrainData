import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from xmlrpc.client import boolean
from keras.models import load_model
import numpy 
import ExcelData as ed
import RegressionModel as rm

def PredictionsConsole(skip:boolean = False):
    """
    Rudimentary console interface for loading a trained model.
    
    Parameters:
    
    skip (boolean): If true the function will skip the user input and use
    the default values.
    
    Returns:
    
    The three paths to: the saved model, the normalisation array, the data to
    be used for the predictions
    """


    if skip:
        model_path = "saves\\2024-09-01 23-22-10.keras"
        nr_path = "saves\\normalisation\\2024-08-22 00-22-46.txt"
        pr_path = "input\\FS_features_ABIDE_males_someGlobals.xlsx" 
        return model_path, nr_path, pr_path

    while True:
            path = input("Please insert the path to the trained model.")
            if os.path.isdir(path):
                skip = "yes"
                break
            else: print("Path is not valid.")        
            
    while True:
            nr_path = input(
                 "Please insert the path to the normalisation array.")
            if os.path.isdir(nr_path): break
            else: print("Path is not valid.")        

    while True:
        pr_path = input("Provide the file to use for predictions.")
        if os.path.isdir(pr_path):
            skip = "yes"
            break
        else: print("Path is not valid.")
    
    return path, nr_path, pr_path


if __name__ == "__main__":
    _ = os.system('cls')
    current_directory = os.getcwd()
 
    model_path ,nr_path,pr_path = PredictionsConsole(True)
    
    samples = ed.ExcelData(pr_path, normalisation = False, shuffle = False)
    samples.Normalisation(numpy.loadtxt(nr_path))
    my_matrix = samples.data_grid
    mlp = load_model(model_path, compile=False)
    predicted_ages = mlp.predict(my_matrix)

    # mlp = rm.RegressionModel(saved_model_path = model_path)
    # predicted_ages = mlp.Predict(my_matrix)
    print(predicted_ages)
