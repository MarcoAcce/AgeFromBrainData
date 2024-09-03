from datetime import datetime
import os
import pandas as pd
import numpy as np

from IData import IData


class ExcelData(IData):
    """Class for handling Excel data."""
    
    def __init__(self, 
                 file_path, 
                 normalisation = False, 
                 shuffle = False):
        """
        Constructor for ExcelData class.

        Parameters:
        
        file_path (str): Path to the Excel file.
        
        normalisation (bool): If true data will be normalized. 
        Default is False.
        
        shuffle (bool): If true rows will be shuffled. Default is False.
        """
        
        self._file_path = file_path
        self._data_frame = None # full data from the .xlsx
        self._numpy_data = None # numpy array of the data
        self._selected_column = None
        
        self._removed_columns = ['FILE_ID', 'AGE_AT_SCAN']
        
        self._used_columns = ['FILE_ID', 'AGE_AT_SCAN', 'SEX','FIQ',
                              'DX_GROUP', 'lh_MeanThickness', 
                              'rh_MeanThickness', 'lhCortexVol', 
                              'rhCortexVol',
                              'lhCerebralWhiteMatterVol',
                              'rhCerebralWhiteMatterVol', 'TotalGrayVol']
        self._load_data(shuffle)
        if normalisation: self.Normalisation()

    @property
    def data_grid(self):
        """The data grid extracted from the input .xlsx file (pandas)."""
        return self._numpy_data

    def _load_data(self, 
                   shuffle = False):
        """
        Load data from the Excel file.
        """
        
        if not os.path.exists(self._file_path):
            raise ValueError("File {} doesn't exist!".format(self._file_path))

        try:
            self._data_frame = pd.read_excel(self._file_path)\
                [self._used_columns]
        
        except Exception as e:
            print(f"Error loading data from {self._file_path}: {e}")
        
        if shuffle : self._shuffle_rows()
        
        self._numpy_data = self._data_frame.drop(
                                        columns=['FILE_ID', 'AGE_AT_SCAN'])\
                               .to_numpy(copy = False, na_value=-9999, 
                                          dtype=float)
        
        self._selected_column = self._data_frame['AGE_AT_SCAN']\
                                    .to_numpy(copy = True, dtype=float)

    def _remove_column(self, 
                       column_name):
        """
        Remove a selected column from the data array. 
        Does not modify the data frame.
                
        Parameters:
        
        column_name (str): Name of the column to select.
        """
        to_remove = self._removed_columns
        to_remove.append(column_name)
        if column_name in self._data_frame.columns:
            self._numpy_data = self._data_frame.drop(
                columns=to_remove)\
               .to_numpy(copy = False, na_value=-9999, dtype=float)
            self._removed_columns.append(column_name)
        else:
            raise ValueError(f"Column '{column_name}' not found",
                             "in data frame.")


    def _shuffle_rows(self):
        """
        Shuffle all rows in the data frame.
        """
        self._data_frame = self._data_frame.sample(frac=1)\
                            .reset_index(drop=True)

    def Save_Normalisation(self):
        """
        Exports the max values of the columns used for training to a 
        .txt file, in a subdirectory "/saves/" of the current dir. 
        The values are the normalisation to be used on new data for the 
        trained model.        
        """
        normalisation_path = os.path.join(os.getcwd(), 
                                          r'saves', r'normalisation')
        os.makedirs(normalisation_path, exist_ok= True)
        name =  str(datetime.now().strftime("%Y-%m-%d %H-%M-%S") ) + ".txt"
        normalisation_path = os.path.join( normalisation_path + name)             
        #write the normalising value to .txt for later use
        with open(normalisation_path, 'w') as file:
            for column in self._data_frame.drop(
                columns = self._removed_columns).columns:
                normalisation = self._data_frame[column].max() 
                file.write(f"{normalisation}\n")
        print(
            f"Normalisation values have been written to {normalisation_path}")

    def Select_column(self, 
                      column_name):
        """
        Get a selected column from the data frame.

        Parameters:
        
        column_name (str): Name of the column to select.
        
        Returns:
        
        numpy.ndarray: Selected column as a numpy array.
        """
        if column_name in self._data_frame.columns:
            self._selected_column = self._data_frame[column_name]\
                                    .to_numpy(copy=True, dtype=float)
            return self._selected_column
        
        else:
            raise ValueError(f"Column '{column_name}' not found in ", 
                             "data frame.")

    def Normalisation(self, 
                      normalisation_array = None):
        """
        Divides all entries in dataframes by the corresponding value
        found in the input array. Skips all entries equal to -9999.
        This method should be used to normalise values before making
        prediction with an already trained array, the normalisation 
        should match the one used during training. 
                
        Parameters:
        
        normalisation_array : numpy array of the values to be used for
        the normalisation. By default will normalise with the maximum
        value for each column.
        
        """
        #checks for an array for external normalisation
        if isinstance(normalisation_array, np.ndarray): 
            external = True
        else: external = False
        data = self.data_grid
        #finds the dimension of the data matrix
        ncolumn = data.shape[1]
        nrows = data.shape[0]
        #checks if the the external normalisation is appropriate for
        #the size of the matrix
        if external and ncolumn != normalisation_array.size:
            raise Exception("normalisation array doesn't match data.")
        #normalises the matrix using external or the max value 
        #for each column. Ignores all -9999 cells.
        for y in range (ncolumn):
            if external : column_normalise = normalisation_array[y]
            else: column_normalise = max(data[:,y])

            for x in range(nrows):
                if data[x,y] == -9999: continue
                else: data[x,y] = data[x,y] / column_normalise
        self._numpy_data = data