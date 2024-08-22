from datetime import datetime
import os
import pandas as pd

from IData import IData


class ExcelData(IData):
    """Class for handling Excel data."""
    
    def __init__(self, file_path, normalisation = False, shuffle = False):
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
        
        self._removed_columns = 'FILE_ID', 'AGE_AT_SCAN'
        
        self._used_columns = ['FILE_ID', 'AGE_AT_SCAN', 'SEX','FIQ',
                              'DX_GROUP', 'lh_MeanThickness', 
                              'rh_MeanThickness', 'lhCortexVol', 'rhCortexVol',
                              'lhCerebralWhiteMatterVol',
                              'rhCerebralWhiteMatterVol', 'TotalGrayVol']
        self._load_data(shuffle)
        if normalisation: self._normalize_columns

    @property
    def data_grid(self):
        """The data grid extracted from the input .xlsx file (pandas)."""
        return self._numpy_data


    def _load_data(self, shuffle = False):
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

    def _remove_column(self, column_name):
        """
        Remove a selected column from the data frame.
                
        Parameters:
        
        column_name (str): Name of the column to select.
        """
        if column_name in self._data_frame.columns:
            
            self._numpy_data = self._data_frame.drop(
                columns=[self._removed_columns, column_name])\
               .to_numpy(copy = False, na_value=-9999, dtype=float)
            
            self._removed_columns.__add__(column_name)
        
        else:
            raise ValueError(f"Column '{column_name}' not found",
                             "in data frame.")
    
    def _normalize_columns(self):
        """
        Divides all entries in dataframes by their column's max value before
        filling the array. Skips all entries equal to -9999.
        """
        ncolumn = self._numpy_data.shape[1]
        nrows =self._numpy_data.shape[0]
        column_min, column_max = 0
        
        for y in range (ncolumn):
            column_max = max(self._numpy_data[:,y])
            column_min = min(self._numpy_data[:,y])
            column_normalise = column_max - column_min
            
            for x in range(nrows):
                if self._numpy_data[x,y] == -9999:
                    continue
                self._numpy_data[x,y] = self._numpy_data[x,y] / column_max

    def _shuffle_rows(self):
        """
        Shuffle all rows in the data frame.
        """
        self._data_frame = self._data_frame.sample(frac=1)\
                            .reset_index(drop=True)

    def Save_Normalisation(self):
        """
        Exports the max values of the columns used for training to a 
        .txt file. The values are the normalisation to be used on new data
        for the trained model.        
        """
        normalisation_path = ("saves/normalisation/" + 
                              str(datetime.now().strftime(
                                  "%Y-%m-%d %H-%M-%S") ) +
                              ".txt")

        with open(normalisation_path, 'w') as file:
            for column in self._data_frame.drop(self._removed_columns).columns:
                normalisation = self._data_frame[column].max() 
                file.write(f"{normalisation}\n")
        print(f"Normalisation values have been written to {normalisation_path}")

    def Select_column(self, column_name):
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

    def External_Normalisation(self, normalisation_array):
        """
        Divides all entries in dataframes by the corresponding value
        found in the input array. Skips all entries equal to -9999.
        This method should be used to normalise values before making
        prediction with an already trained array, the normalisation 
        should match the one used during training.
        
        Parameters:
        
        normalisation_array : numpy array of the values to be used for
        the normalisation
        """
        ncolumn = self._numpy_data.shape[1]
        nrows =self._numpy_data.shape[0]
        
        if ncolumn != normalisation_array.size: 
            raise ValueError(f"Normalisation array not of the required size ", 
                             "for the data frame.")
        
        self._load_data(shuffle=False)

        for y in range (ncolumn):
            column_normalise = normalisation_array[y]
            
            for x in range(nrows):
                if self._numpy_data[x,y] == -9999:
                    continue
                self._numpy_data[x,y] = self._numpy_data[x,y] / column_normalise
