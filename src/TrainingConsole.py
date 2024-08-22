import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
import ExcelData as ed
import RegressionModel as rm

def TrainingConsole(X, y):
    """Function providing a command line UI for the model."""
    print("Welcome to the Regression Model Training Console App!")
 
    #default values for the model
    resume_training = "no"
    epoch_number = 100000
    number_of_hidden_layers = 2
    number_of_hidden_nodes = 6
    adv_sett = "no"
    patience = 2000
    

    # Prompt the user for input
    while True:
        skip = input("Do you want to train the model using the default", 
                     "  parameters? (yes/no): ").strip().lower() 
        if skip == "yes" or skip == "no":
            break
        else:
            print("Please only enter yes or no.")
    
    while skip == "no":
        resume_training = input("Do you want to resume training" ,
                                "from a checkpoint?  (yes/no): "
                                ).strip().lower() 
        if resume_training == "yes" or resume_training == "no":
            break
    
    checkpoint = None
    if resume_training == "yes":
        while skip == "no":
            checkpoint = input("Enter the path to resume from ",
                               "a checkpoint: ").strip()
            if not checkpoint:
                print("Error:", 
                      "You chose to resume training but did not provide a",  
                      "checkpoint path.\n")
                continue
            if not os.path.exists(checkpoint):
                print("Error: Checkpoint file not found.\n")
                continue
            else:
                break
    else:
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
            break
        except ValueError:
            print("Please insert a valid number.\n")

    while skip == "no":
        adv_sett = input("Do you want to modify validation split",
                         "and patience? (yes/no):"
                         ).strip().lower()
        if adv_sett == "yes" or adv_sett == "no":
            break
        else:
            print("Please only eneter yes or no.")
            continue

    mlp = rm.RegressionModel(hidden_layers=number_of_hidden_layers, 
                             nodes_per_layer=number_of_hidden_nodes)
    
    mlp.epoch_number = epoch_number
    
    mlp.Compile_Model(X.shape[1], )
    
    # Training model ---

    while adv_sett == "yes":
        patience = input("Please enter the patience value").strip()
        split = input("Please enter the validation split value").strip()
        try:
            patience = int(patience)
            split = int(split)
            break
        except (TypeError, ValueError):
            print("Error: please input valid int numbers.")
            continue
  
    if resume_training == "yes":
        history = mlp.Resume_Training(X, y, 
                                      from_checkpoint_path = checkpoint, 
                                      epochs=epoch_number)
    else:
        if adv_sett == "yes":
            history = mlp.Start_Training(X,y, epochs=epoch_number,
                                         stopping_patience=patience,
                                         validation_split=split)
        else: history = mlp.Start_Training(X,y, epochs=epoch_number)

    # Saving trained model ---

    mlp.Save_Model()
    #ed.Save_Normalisation()
    mlp.Plot_History()
    
    
    
        


if __name__ == "__main__":
    _ = os.system('cls')
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 
                             r'input\FS_features_ABIDE_males_someGlobals.xlsx')
    samples = ed.ExcelData(file_path, True)
    my_matrix = samples.data_grid
    age = samples.Select_column('AGE_AT_SCAN')
    TrainingConsole(my_matrix, age)