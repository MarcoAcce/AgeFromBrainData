import pandas as pd
import os

def SplitExcel(file_path: str, columnForSplitting: int =0):
    """
    Function to split an .xlsx file into multiple files, grouping together
    all rows which share the same key in a column.
    
    Parameters:
    
    file_path (str): path to the file to be splt.
    
    columnForSplitting (int): index of the column to be used for grouping 
    (starts at 0).
    
    """
    # Load the original .xlsx file
    df = pd.read_excel(input_file)
    
    # Group data by the name in the first column
    #Will take only the substring before the "_" char.
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



if __name__ == 'main':
    
    current_directory = os.getcwd()
    input_file = os.path.join(
        current_directory, r'input\FS_features_ABIDE_males_someGlobals.xlsx')
    SplitExcel(input_file,0)
    
    
