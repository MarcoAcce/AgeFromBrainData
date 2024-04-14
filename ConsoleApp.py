import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy
import ExcelData as ed
import RegressionModel as rm

def ConsoleApp(X, y):
    print("Welcome to the Regression Model Training Console App!")
 
    # Prompt the user for input
    """
    while True:
        resume_training = input("Do you want to resume training from a checkpoint? (yes/no): ").lower() #- resume not yet implemented
        if resume_training == "yes" or resume_training == "no":
            break
    if resume_training == "yes":
        while True:
            checkpoint_path = input("Enter the path to resume from a checkpoint: ").strip()
            if not checkpoint_path:
                print("Error: You chose to resume training but did not provide a checkpoint path.\n")
                continue
            if not os.path.exists(checkpoint_path):
                print("Error: Checkpoint file not found.\n")
                continue
            else:
                break
    """    
    while True:
        try: 
            epoch_number = int(input("Enter the number of epochs for training: "))
            break
        except ValueError:
            print("Please insert a valid number.\n")
    while True:
        adv_sett = input("Do you want to modify validation split and patience? (yes/no): ")
        if adv_sett == "yes" or adv_sett == "no":
            break

    

    mlp = rm.RegressionModel(hidden_layers=2, nodes_per_layer=64, patience=5000)
    mlp.Define_Model(X.shape[1])  # You may need to modify this method to take shape as input
    #if resume_training == "yes":
    #   history = mlp.Resume_Training(X, y, from_checkpoint_path = checkpoint_path, epochs=epoch_number)
    #else:
    history = mlp.Start_Training(X,y, epochs=epoch_number)
    mlp.Plot_History()
        


if __name__ == "__main__":
    _ = os.system('cls')
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'input\FS_features_ABIDE_males.xlsx')
    samples = ed.ExcelData(file_path)
    my_matrix = samples.data_grid
    age = samples.select_column('AGE_AT_SCAN')
    ConsoleApp(my_matrix, age)