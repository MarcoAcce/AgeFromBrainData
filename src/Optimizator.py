import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ExcelData as ed
import RegressionModel as rm

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

    for layers in possible_hidden_layers_number:
        for nodes in possible_nodes_number:
            
            if layers*nodes > 14: continue
            
            print(f"Starting | layers: {layers}, nodes: {nodes}")
            mlp = rm.RegressionModel(hidden_layers=layers,
                                     nodes_per_layer=nodes, patience=patience)
            mlp.Compile_Model(my_matrix.shape[1], 
                              number_of_epochs_per_print = 100)
            
            history = mlp.Start_Training(my_matrix, age, epochs=epoch_number)
            
            print(f"Finished | layers: {layers}, nodes: {nodes}")
            
            

