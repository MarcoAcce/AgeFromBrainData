import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from ExcelData import ExcelData

class TestExcelData(unittest.TestCase):

    def setUp(self):
        # Mock data for testing
        self.mock_data = {
            'FILE_ID': [1, 2, 3],
            'AGE_AT_SCAN': [25, 35, 45],
            'SEX': [1, 0, 1],
            'FIQ': [110, 120, 130],
            'DX_GROUP': [1, 2, 1],
            'lh_MeanThickness': [2.5, 3.0, 2.8],
            'rh_MeanThickness': [2.6, 3.1, 2.9],
            'lhCortexVol': [40000, 42000, 41000],
            'rhCortexVol': [40500, 42500, 41500],
            'lhCerebralWhiteMatterVol': [30000, 32000, 31000],
            'rhCerebralWhiteMatterVol': [30500, 32500, 31500],
            'TotalGrayVol': [90000, 95000, 92000]
        }
        self.mock_df = pd.DataFrame(self.mock_data)
        
        # Mock the path and dataframe loading
        self.mock_file_path = 'mock_path.xlsx'
        self.patcher = patch('pandas.read_excel', return_value=self.mock_df)
        self.mock_read_excel = self.patcher.start()

        self.excel_data = ExcelData(self.mock_file_path)

    def tearDown(self):
        self.patcher.stop()

    def test_load_data(self):
        # Verify that data is loaded correctly
        self.assertTrue(np.array_equal(self.excel_data._numpy_data,
                                       self.mock_df.drop(columns=['FILE_ID', 'AGE_AT_SCAN']).to_numpy(dtype=float)))

        self.assertTrue(np.array_equal(self.excel_data._selected_column,
                                       self.mock_df['AGE_AT_SCAN'].to_numpy(dtype=float)))

    def test_remove_column(self):
        # Test removing a column
        self.excel_data._remove_column('SEX')
        self.assertNotIn('SEX', self.excel_data._data_frame.columns)

        with self.assertRaises(ValueError):
            self.excel_data._remove_column('NonExistentColumn')

    def test_normalization(self):
        # Test normalization without passing an array (using max values)
        original_data = self.excel_data._numpy_data.copy()
        self.excel_data._normalize_columns()

        for i, col in enumerate(self.mock_df.drop(columns=['FILE_ID', 'AGE_AT_SCAN']).columns):
            self.assertTrue(np.allclose(self.excel_data._numpy_data[:, i],
                                        original_data[:, i] / max(original_data[:, i])))

    def test_shuffle_rows(self):
        # Test if shuffling changes the order of rows
        original_data = self.excel_data._data_frame.copy()
        self.excel_data._shuffle_rows()

        self.assertFalse(original_data.equals(self.excel_data._data_frame))

    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('builtins.open')
    def test_save_normalisation(self, mock_open, mock_makedirs, mock_exists):
        # Test saving normalization
        self.excel_data.Save_Normalisation()
        mock_open.assert_called_once()

    def test_select_column(self):
        # Test selecting a column
        selected_column = self.excel_data.Select_column('SEX')
        self.assertTrue(np.array_equal(selected_column, self.mock_df['SEX'].to_numpy(dtype=float)))

        with self.assertRaises(ValueError):
            self.excel_data.Select_column('NonExistentColumn')

    def test_normalisation_with_array(self):
        # Test normalization with a provided normalization array
        norm_array = np.array([1.0, 100.0, 1000.0, 10000.0, 100.0, 10000.0, 100000.0, 10000.0, 100000.0, 1000000.0])

        self.excel_data.Normalisation(normalisation_array=norm_array)

        for i in range(self.excel_data._numpy_data.shape[1]):
            for j in range(self.excel_data._numpy_data.shape[0]):
                if self.mock_df.iloc[j, i+2] != -9999:  # Offset by 2 due to dropped columns
                    expected_value = self.mock_df.iloc[j, i+2] / norm_array[i]
                    self.assertAlmostEqual(self.excel_data._numpy_data[j, i], expected_value)

if __name__ == '__main__':
    unittest.main()
