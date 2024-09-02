import pandas as pd
from os import getcwd, path, makedirs

def SplitExcel(file_path: str, columnForSplitting: int =0,
               output_directory_path: str = None):
    """
    Function to split an .xlsx file into multiple files, grouping together
    all rows which share the same key in a column.
    
    Parameters:
    
    file_path (str): path to the file to be splt.
    
    columnForSplitting (int): index of the column to be used for grouping 
    (starts at 0).
    
    output_directory_path (str): path to the directory to which the output 
    will be printed. Default at None, will use the working directory
    
    """
    if not path.isfile(file_path): 
        raise Exception("file at ",file_path, "not found!")

    # Load the original .xlsx file
    df = pd.read_excel(file_path)
    
    # Group data by the name in the first column
    #Will take only the substring before the "_" char.
    df['GroupKey'] = df[df.columns[0]].str.split('_').str[0]
    grouped = df.groupby('GroupKey')
    
    if output_directory_path == None:
        output_directory_path = getcwd()
    #creates a subdirectory "split" of the directory    
    output_directory_path = path.join(output_directory_path, r'split')
    makedirs(output_directory_path, exist_ok = True)

       
    # Loop through each group and write to separate .xlsx files
    for key, group in grouped:
        group = group.drop(columns=['GroupKey'])
        # Create a new filename based on the value in the first column
        output_file = path.join(
            output_directory_path, f"{key}.xlsx" )
        
        # Save the group to a new .xlsx file, including the column titles
        group.to_excel(output_file, index=False)
        
        print(f"Data for '{key}' written to {output_file}")



if __name__ == '__main__':
        
    current_directory = getcwd()
    
    input_file = path.join(
        current_directory,r'input\FS_features_ABIDE_males_someGlobals.xlsx')
    
    SplitExcel(input_file,0, path.join(current_directory, r"input"))
    
    
