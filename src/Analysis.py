from os import listdir, getcwd
from os.path import isfile, join
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import SplitExcel as se

"""
"Script for the data analysis of the predicted ages.
Splits the model predictions for the entire data set on smaller
.xlsx files for each data group, then calculates the RMS and MSE
of the prediction for each group.
The result are saved in a plot, shown against the resulting RMS and
MSE for the complete data set.
"""

def Analysis (path_input_file: str, analysis_directory: str = None):
    
    #split the model_predictions 
    try: se.SplitExcel(path_input_file, 0, analysis_directory)
    except Exception as ex: 
        print(ex)
        quit()
    #select all files split from model_predictions
    #listdir returns only the files' name and not the paths
    #the split removes the file's extension    
    analysis_split_directory = join(analysis_directory, r'split')
    analysis_files_names = [f.split(".xlsx")[0] 
                            for f in listdir(analysis_split_directory) 
                            if isfile(join(analysis_split_directory, f))]
    #creates empty dictionaries for the results 
    rms_list = []
    mse_list = []
    total_mse = 0
    total_patient_number = 0
    
    #iterate through all split files
    for file_name in analysis_files_names:
        #reconstruct the path to each file
        file = join(analysis_split_directory, file_name + ".xlsx")
        #reads age and predictions for the file
        df = pd.read_excel(file)[['AGE_AT_SCAN','PREDICTED_AGE']]
        ages = df['AGE_AT_SCAN'].to_numpy(dtype=float)
        predictions = df['PREDICTED_AGE'].to_numpy(dtype=float)
        #sanity check for the arrays' dimensions
        if ages.size != predictions.size:
            raise Exception("Ages and predicted ages differ in number!")
    
        #calculate mse and rms for each group and save value to a dictionary
        mse = 0
        rms = 0
    
        for age, prediction in zip(ages,predictions):
            square_error = (age - prediction) ** 2
            mse += square_error
            rms += sqrt(square_error)
            total_mse += square_error
        mse = mse / ages.size  
        rms = rms / ages.size
        rms_list.append(rms)
        mse_list.append(mse)
        total_patient_number += ages.size
        print(file_name, "   rms: ", rms, "   mse: ", mse)
    
    total_mse = total_mse / total_patient_number
    total_rms = sqrt(total_mse)
    print("Complete dataset", "   rms: ", total_rms, "   mse :", total_mse)
    
    #create two subplots for mse and rms
    fig, axs = plt.subplots(2, sharex=True, figsize=(14, 10))
    rms_plot = axs[0]
    mse_plot = axs[1]
    
    #rms subplot
    rms_plot.plot(analysis_files_names,rms_list)
    #array of all the y value for the grid
    y_rms_ticks = np.arange(0, max(rms_list) + 1, 1)
    #grid
    rms_plot.set_yticks(y_rms_ticks)  
    rms_plot.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    #horizontal line at the total value
    rms_plot.axhline(total_rms, color='black', 
                     linestyle=':', label=f'Complete data RMS: {total_rms:.2f}')
    rms_plot.set(xlabel = 'Group', ylabel = 'RMS [yr]')
    #legend
    rms_plot.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    #mse subplot
    mse_plot.plot(analysis_files_names,mse_list)
    #array of all the y value for the grid
    y_mse_ticks = np.arange(0, max(mse_list) + 20, 20)
    #grid
    mse_plot.set_yticks(y_mse_ticks)     
    mse_plot.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    #horizontal line at the total value
    mse_plot.axhline(total_mse, color='black', 
                     linestyle=':', label=f'Complete data MSE: {total_mse:.2f}')
    mse_plot.set(xlabel = 'Group', ylabel = 'MSE')
    #legend
    mse_plot.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    #group name labels rotation
    plt.xticks(rotation=45, ha='right')
    #save plot as result.png
    plt.savefig(analysis_directory + "\\result.png", dpi=500, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    analysis_directory = join(getcwd(),r'analysis')
    path_input_file = join(
        analysis_directory, f'model_predictions.xlsx')
    Analysis(path_input_file, analysis_directory)