import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ExcelData as ed
import RegressionModel as rm
from PredictionsConsole import PredictionsConsole

_ = os.system('cls')
current_directory = os.getcwd()
file_path = os.path.join(current_directory, 
                         r'input\FS_features_ABIDE_males_someGlobals.xlsx')
samples = ed.ExcelData(file_path, False)
my_matrix = samples.data_grid
age = samples.Select_column('AGE_AT_SCAN')

mlp = rm.RegressionModel()

epoch_number = 10000
number_of_hidden_layers = 2
number_of_hidden_nodes = 6
split = 0.8
adv_sett = "no"
patience = 1000

mlp.epoch_number = epoch_number

mlp.Compile_Model(my_matrix.shape[1],hidden_layers=number_of_hidden_layers,
                  nodes_per_layer=number_of_hidden_nodes)
                  
mlp.Start_Training(my_matrix,age, epochs=epoch_number,
               stopping_patience=patience,
               validation_split=split)                  
mlp.Save_Model()
#ed.Save_Normalisation()

mlp.Plot_History()
model_path ,nr_path,pr_path = PredictionsConsole(True)

#samples = ed.ExcelData(pr_path, False)
#samples.External_Normalisation(numpy.loadtxt(nr_path))
#my_matrix = samples.data_grid

predicted_ages = mlp.Predict(my_matrix)
print(predicted_ages)