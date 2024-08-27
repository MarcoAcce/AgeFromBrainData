import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ExcelData as ed
import RegressionModel as rm

def Optimizator(X, y,
                possible_hidden_layers_number,
                possible_nodes_number):
    """
    Function to cycles through a number of possible model configurations
    and train for a limited time, to decide on the best configuration for
    the final model.
    
    Parameters:
    X (ndarray): Numpy array containg the featurs for model training
    
    y (ndarray): Numpy array of the samples' labels for training.
    
    possible_hidden_layers_number: Array of list of integer numbers, 
    containing all possible numbers of hidden layers of the model.
    
    possible_nodes_number: Array of list of integer numbers, 
    containing all possible numbers of nodes for the hidden layers 
    of the model.
    """
    for layers in possible_hidden_layers_number:
        for nodes in possible_nodes_number:
            
            if layers*nodes > 14: continue
            
            print(f"Starting | layers: {layers}, nodes: {nodes}")
            mlp = rm.RegressionModel(hidden_layers=layers,
                                     nodes_per_layer=nodes, patience=patience)
            mlp.Compile_Model(X.shape[1], 
                              number_of_epochs_per_print = 100)
            
            history = mlp.Start_Training(X, y, epochs=epoch_number)
            
            print(f"Finished | layers: {layers}, nodes: {nodes}")

if __name__ == "__main__":
    _ = os.system('cls')
    print("Starting optimization cycle:\n")
    patience = 1000
    epoch_number = 10000
    
    current_directory = os.getcwd()
    file_path = os.path.join(
        current_directory, r'input\FS_features_ABIDE_males.xlsx')
    
    samples = ed.ExcelData(file_path, True)
    my_matrix = samples.data_grid
    age = samples.Select_column('AGE_AT_SCAN')
    
    possible_hidden_layers_number = (1, 2, 3)
    possible_nodes_number = (3, 4, 5, 6, 7)

    Optimizator(my_matrix, age, 
                possible_hidden_layers_number,possible_nodes_number)

            
            

