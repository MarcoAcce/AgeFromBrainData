import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy 
import ExcelData as ed
import RegressionModel as rm

def PredictionsConsole(skip = False):
    
    if skip:
        model_path = "saves\2024-08-05 23-40-24.keras"
        nr_path = "saves\\normalisation\\2024-08-22 00-22-46.txt"
        pr_path = "input\FS_features_ABIDE_males.xlsx" 
        return model_path, nr_path, pr_path

    while True:
            path = input("Please insert the path to the trained model.")
            if os.path.isdir(path):
                skip = "yes"
                break
            else: print("Path is not valid.")        
            
    while True:
            nr_path = input("Please insert the path to the normalisation array.")
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
    
    samples = ed.ExcelData(pr_path, False)
    samples.External_Normalisation(numpy.loadtxt(nr_path))
    my_matrix = samples.data_grid
    
    mlp = rm.RegressionModel(saved_model_path = model_path)
    predicted_ages = mlp.Predict(my_matrix)
    print(predicted_ages)
