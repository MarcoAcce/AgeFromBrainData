import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from RegressionModel import RegressionModel

def testRegressionModel(test_compile=False,
                        test_train=False,
                        test_predict=False,
                        test_save_model=False,
                        test_plot_history=False):
    
    # Initialize the RegressionModel object
    regression_model = RegressionModel()

    # Test compiling the model
    if test_compile:
        input_shape = 10  # input shape
        regression_model.Compile_Model(shape=input_shape, hidden_layers=3, nodes_per_layer=64)
        print("Model compiled successfully.")
        regression_model._model.summary()

    # Test training the model
    if test_train:
        X_train = np.random.rand(100, 10)  # training data
        y_train = np.random.rand(100)      # label data
        regression_model.Compile_Model(shape=X_train.shape[1], hidden_layers=3, nodes_per_layer=64)
        history = regression_model.Start_Training(X=X_train, y=y_train, epochs=10, validation_split=0.2)
        print("Model trained successfully.")
        print("Training history:", history.history)

    # Test predicting with the model
    if test_predict:
        X_test = np.random.rand(10, 10)  # fake data
        predictions = regression_model.Predict(X_test)
        print("Predictions:", predictions)

    # Test saving the model
    if test_save_model:
        regression_model.Save_Model()
        print("Model saved successfully.")

    # Test plotting the training history
    if test_plot_history:
        regression_model.Plot_History()
        print("Training history plot saved.")
        print(regression_model._save_path)

if __name__ == '__main__':
    # Test the RegressionModel class
    testRegressionModel(test_compile=True,
                        test_train=True,
                        test_predict=True,
                        test_save_model=True,
                        test_plot_history=True)
