How to use this code
====================

This project provides a classes and script to train a regression model on brain MRI scans data to predict the patient's age.

The model, based on the Keras library, is implemented in the RegressionModel class:
   
	The model is a deep neural network regressor, it expects a numpy array of features (mri data) and an array of labels (ages) for supervised training.
	The RegressionModel class provides methods to define and compile the model, train over data, save and load trained models and make predictions.

The project provides a ExcelData class to process .xlsx files of features and labels into the array expected by the model.
	The ExcelData class includes methods to select the labels' coloumn from the dataframe, to remove unnecessary features, and to normalise and shuffle the data.
	For the bext results the data should be normalised and shuffled before training, all the data for predictions should be normalised in the same way as the training data and not shuffled.

The project includes two functions for basic UI in the command console: one for model training, TrainingConsole, and one for making predictions with pre-trained model, PredictionsConsole.
Both of these functions allow the user to skip all input to use default values set in the source code.

Finally the project provides some utility scripts to evaluate the model's predictions and to choose the best hypeparameters.