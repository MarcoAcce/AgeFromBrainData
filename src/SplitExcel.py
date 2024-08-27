import pandas as pd
import os

if __name__ == 'main':
    
    current_directory = os.getcwd()
    input_file = os.path.join(
        current_directory, r'input\FS_features_ABIDE_males_someGlobals.xlsx')
    
    # Load the original .xlsx file
    df = pd.read_excel(input_file)
    
    # Group data by the name in the first column
    df['GroupKey'] = df[df.columns[0]].str.split('_').str[0]
    grouped = df.groupby('GroupKey')
    
    # Loop through each group and write to separate .xlsx files
    for key, group in grouped:
        group = group.drop(columns=['GroupKey'])
        # Create a new filename based on the value in the first column
        output_file = os.path.join(
            current_directory, r'input\split', f"{key}.xlsx" )
        
        # Save the group to a new .xlsx file, including the column titles
        group.to_excel(output_file, index=False)
        
        print(f"Data for '{key}' written to {output_file}")
    
