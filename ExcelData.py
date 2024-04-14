import os
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
    def __init__(self, file_path):
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
        self._load_data()

    def _load_data(self):

        if not os.path.exists(self._file_path):
            raise ValueError("File {} doesn't exist!".format(self._file_path))

        """Load data from the Excel file."""
        try:
            self._data_frame = pd.read_excel(self._file_path)
        except Exception as e:
            print(f"Error loading data from {self._file_path}: {e}")
        self._numpy_data = numpy.array(self._data_frame.drop(columns=['FILE_ID', 'AGE_AT_SCAN']), float)
        self._selected_column = numpy.array(self._data_frame['AGE_AT_SCAN'], float)

    @property
    def data_grid(self):
        """Get the data grid."""
        return self._numpy_data

    def remove_column(self, column_name):
        """
        Remove a selected column from the data frame.

        Parameters:
        column_name (str): Name of the column to select.
        """
        if column_name in self._data_frame.columns:
            self._numpy_data = numpy.array(self._data_frame.drop(columns=[self._removed_columns, column_name]), float)
        else:
            raise ValueError(f"Column '{column_name}' not found in the data frame.")
    
    def select_column(self, column_name):
        """
        Get a selected column from the data frame.

        Parameters:
        column_name (str): Name of the column to select.
        
        Returns:
        numpy.ndarray: Selected column as a numpy array.
        """
        if column_name in self._data_frame.columns:
            self._selected_column = numpy.array(self._data_frame[column_name], float)
            return self._selected_column
        else:
            raise ValueError(f"Column '{column_name}' not found in the data frame.")

class AnotherData(IData):
    """Class for handling another type of data."""
    def __init__(self, data_grid, selected_column):
        """
        Constructor for AnotherData class.

        Parameters:
        data_grid (numpy.ndarray): The data grid.
        selected_column (numpy.ndarray): The selected column.
        """
        self._data_grid = data_grid
        self._selected_column = selected_column

    @property
    def data_grid(self):
        """Get the data grid."""
        return self._data_grid

    @property
    def selected_column(self):
        """Get the selected column."""
        return self._selected_column
