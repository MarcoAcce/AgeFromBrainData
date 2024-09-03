import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest

import numpy as np
from unittest.mock import patch, MagicMock
from RegressionModel import RegressionModel 

class TestRegressionModel(unittest.TestCase):

    def setUp(self):
        # Create a basic instance of RegressionModel without a saved model
        self.model = RegressionModel()

    def test_compile_model(self):
        # Test the Compile_Model method
        shape = 10
        hidden_layers = 2
        nodes_per_layer = 5

        self.model.Compile_Model(shape, hidden_layers, nodes_per_layer)

        # Check if the model is not None after compilation
        self.assertIsNotNone(self.model._model)

        # Check if the model has the correct number of layers
        self.assertEqual(len(self.model._model.layers), hidden_layers + 2)  # input + hidden + output

    def test_start_training_without_model(self):
        # Test Start_Training without compiling the model (should raise ValueError)
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 1)

        with self.assertRaises(ValueError):
            self.model.Start_Training(X, y)

    @patch('RegressionModel.Model.fit')
    def test_start_training(self, mock_fit):
        # Test Start_Training with a compiled model
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 1)

        # Mock the model fitting process
        mock_fit.return_value = MagicMock(history={'loss': [], 'val_loss': []})
        self.model.Compile_Model(X.shape[1], hidden_layers=2, nodes_per_layer=5)
        history = self.model.Start_Training(X, y, validation_split=0.25, epochs=10)

        # Check if the history is returned
        self.assertIsNotNone(history)

        # Verify that fit was called with the correct parameters
        mock_fit.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_history(self, mock_savefig, mock_show):
        # Test Plot_History method

        # Mock the training history
        self.model._history = MagicMock(history={'loss': [1, 0.8, 0.6], 'val_loss': [1.2, 1.0, 0.9]})

        self.model.Plot_History()

        # Check if the plot was saved
        mock_savefig.assert_called_once()

        # Check if the plot was shown
        mock_show.assert_called_once()

    @patch('keras.models.Model.save')
    @patch('keras.models.Model.save_weights')
    def test_save_model(self, mock_save, mock_save_weights):
        # Test Save_Model method

        self.model.Compile_Model(shape=10, hidden_layers=2, nodes_per_layer=5)
        self.model.Save_Model()

        # Check if the model save method was called
        mock_save.assert_called_once()

        # Check if the save_weights method was called
        mock_save_weights.assert_called_once()

    @patch('keras.models.Model.predict')
    def test_predict(self, mock_predict):
        # Test Predict method
        X = np.random.rand(10, 10)
        mock_predict.return_value = np.random.rand(10, 1)

        self.model.Compile_Model(shape=10, hidden_layers=2, nodes_per_layer=5)
        predictions = self.model.Predict(X)

        # Check if the predict method was called
        mock_predict.assert_called_once_with(X)

        # Check if predictions were returned
        self.assertIsNotNone(predictions)

    def test_load_model(self):
        # Test _load_model method with an invalid path
        with self.assertRaises(ValueError):
            self.model._load_model('invalid_model.txt')

        # Test with valid path (assuming .keras file, but using a mock)
        with patch('keras.models.load_model') as mock_load_model:
            self.model._load_model('valid_model.keras')
            mock_load_model.assert_called_once_with('valid_model.keras', compile=False)

if __name__ == '__main__':
    unittest.main()
