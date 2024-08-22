import pandas as pd
import os
from datetime import datetime
 
current_directory = os.getcwd()
file_path = os.path.join(current_directory, 
                         r'input\FS_features_ABIDE_males_someGlobals.xlsx')
used_columns = [ 'SEX','FIQ',
                    'DX_GROUP', 'lh_MeanThickness',
                    'rh_MeanThickness', 'lhCortexVol', 'rhCortexVol',
                    'lhCerebralWhiteMatterVol',
                    'rhCerebralWhiteMatterVol', 'TotalGrayVol']
data = pd.read_excel(file_path)[used_columns]
# Open a text file to write the output
normalisation_path = ("saves/normalisation/" + 
                      str(datetime.now().strftime(
                          "%Y-%m-%d %H-%M-%S") ) +
                      ".txt")

with open(normalisation_path, 'w') as file:
# Loop through each column in the DataFrame
    for column in data.columns:
    # Get the maximum value of the column
        normalisation = data[column].max()
        # Write the column name and its maximum value to the file
        file.write(f"{normalisation}\n")
print(f"Normalisation values have been written to {normalisation_path}")