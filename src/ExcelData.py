import os
from keras.src.random import normal
import pandas as pd
import numpy


class IData:
    """Interface for data classes."""
    @property
    def data_grid(self):
        """Abstract property representing the data grid."""
        raise NotImplementedError("Subclasses must implement data_grid property")

    @property
    def selected_column(self):
        """Abstract property representing the selected column."""
        raise NotImplementedError("Subclasses must implement selected_column property")


class ExcelData(IData):
    """Class for handling Excel data."""
    def __init__(self, file_path, normalisation = False):
        """
        Constructor for ExcelData class.

        Parameters:
        file_path (str): Path to the Excel file.
        """
        self._file_path = file_path
        self._data_frame = None # full data from the .xlsx
        self._numpy_data = None # numpy array of the data
        self._selected_column = None
        self._removed_columns = 'FILE_ID', 'AGE_AT_SCAN'
        self._used_columns = ['FILE_ID', 'AGE_AT_SCAN', 'SEX','FIQ', 'DX_GROUP', 'lh_MeanThickness', 'rh_MeanThickness', 'lhCortexVol', 'rhCortexVol', 'lhCerebralWhiteMatterVol', 'rhCerebralWhiteMatterVol', 'TotalGrayVol']
        self._load_data()
        if (normalisation):
            self._normalize_columns

    def _load_data(self):

        if not os.path.exists(self._file_path):
            raise ValueError("File {} doesn't exist!".format(self._file_path))

        """Load data from the Excel file."""
        try:
            self._data_frame = pd.read_excel(self._file_path)[self._used_columns]
        except Exception as e:
            print(f"Error loading data from {self._file_path}: {e}")
        self._shuffle_rows()
        self._numpy_data = self._data_frame.drop(columns=['FILE_ID', 'AGE_AT_SCAN']).to_numpy(copy = False, na_value=-9999, dtype=float)
        self._selected_column = self._data_frame['AGE_AT_SCAN'].to_numpy(copy = True, dtype=float)


    @property
    def data_grid(self):
        """Get the data grid."""
        return self._numpy_data

    def _remove_column(self, column_name):
        """
        Remove a selected column from the data frame.

        Parameters:
        column_name (str): Name of the column to select.
        """
        if column_name in self._data_frame.columns:
            self._numpy_data = self._data_frame.drop(columns=[self._removed_columns, column_name]).to_numpy(copy = False, na_value=-9999, dtype=float)
            self._removed_columns.__add__(column_name)
        else:
            raise ValueError(f"Column '{column_name}' not found in the data frame.")
    
    def _normalize_columns(self):
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
                self._numpy_data[x,y] = self._numpy_data[x,y] / column_normalise

    def _shuffle_rows(self):
        self._data_frame = self._data_frame.sample(frac=1).reset_index(drop=True)

    def Select_column(self, column_name):
        """
        Get a selected column from the data frame.

        Parameters:
        column_name (str): Name of the column to select.
        
        Returns:
        numpy.ndarray: Selected column as a numpy array.
        """
        if column_name in self._data_frame.columns:
            #self._selected_column = numpy.array(self._data_frame[column_name], float)
            self._selected_column = self._data_frame[column_name].to_numpy(copy=True, dtype=float)
            return self._selected_column
        else:
            raise ValueError(f"Column '{column_name}' not found in the data frame.")

