import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ExcelData as ed
import RegressionModel as rm

def TrainingConsole(X, y):
    """Function providing a command line UI for the model."""
    print("Welcome to the Regression Model Training Console App!")
 
    #default values for the model
    epoch_number = 10000
    number_of_hidden_layers = 2
    number_of_hidden_nodes = 6
    split = 0.8
    adv_sett = "no"
    patience = 1000
    

    # Prompt the user for input
    while True:
        skip = input(
            "Do you want to train using the default parameters? (yes/no): "
            ).strip().lower() 
        if skip == "yes" or skip == "no":
            break
        else:
            print("Please only enter yes or no.")
    
    while skip == "no":
        number_of_hidden_nodes = input("Enter the number of hidden nodes:"
                                       ).strip()
        try:
            number_of_hidden_nodes = int(number_of_hidden_nodes)
            break
        except (TypeError, ValueError):
            print("Error: please input a valid number of nodes.")
            continue
    
    while skip == "no":
        number_of_hidden_layers = input(
                                "Enter the number of hidden layers:"
                                ).strip()
        try:
            number_of_hidden_layers = int(number_of_hidden_layers)
            break
        except (TypeError, ValueError):
            print("Error: please input a valid number of layers.")
            continue                       

    while skip == "no":
        try: 
            epoch_number = int(input(
                "Enter the number of epochs for training: "))
            if epoch_number < 1 : raise ValueError
            break
        except ValueError:
            print("Please insert a valid number.\n")

    while skip == "no":
        adv_sett = input(
            "Do you want to modify validation split and patience? (yes/no):"
                         ).strip().lower()
        if adv_sett == "yes" or adv_sett == "no":
            break
        else:
            print("Please only eneter yes or no.")
            continue

    mlp = rm.RegressionModel()
    
    mlp.epoch_number = epoch_number
    
    mlp.Compile_Model(X.shape[1],hidden_layers=number_of_hidden_layers,
                      nodes_per_layer=number_of_hidden_nodes)
    
    # Training model ---

    while adv_sett == "yes":
        patience = input("Please enter the patience value: ").strip()
        split = input("Please enter the validation split value: ").strip()
        try:
            patience = int(patience)
            split = float(split)
            break
        except (TypeError, ValueError):
            print("Error: please input valid numbers.")
            continue

    mlp.Start_Training(X,y, epochs=epoch_number,
                       stopping_patience=patience,
                       validation_split=split)

    # Saving trained model ---

    if input(
        "Do you want to save the model: (yes/*)"
        ) == "yes":
        mlp.Save_Model()
        #ed.Save_Normalisation()
        mlp.Plot_History()
    return mlp
    

if __name__ == "__main__":
    _ = os.system('cls')
    current_directory = os.getcwd()
    file_path = os.path.join(
        current_directory,r'input\FS_features_ABIDE_males_someGlobals.xlsx')
    samples = ed.ExcelData(file_path, True, True)
    my_matrix = samples.data_grid
    age = samples.Select_column('AGE_AT_SCAN')
    model = TrainingConsole(my_matrix, age)
    
    samples = ed.ExcelData(file_path, normalisation = True, shuffle = False)

    predicted_ages = model._model.predict(my_matrix)
    print(predicted_ages)
