import os
import signal
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import datetime
from ExcelData import IData,ExcelData
from datetime import datetime

class IRegression:
    """Interface for regression classes."""
    
    def Compile_Model(self, shape=None, checkpoint_path = None):
        raise NotImplementedError("Subclasses must implement Compile_Model method")

    def Start_Training(self, X, y, save_checkpoint_path = None
                       , validation_split = 0.75, epochs = None, verbose_training = 0):
        raise NotImplementedError("Subclasses must implement Start_Training method")

    def Plot_History(self):
        raise NotImplementedError("Subclasses must implement Plot_History method")

class PrintProgress(Callback):
    """Callback for printing progress during model training."""
    _number_of_epochs_per_print=3
    @property
    def number_of_epochs_per_print(self):
        return self._number_of_epochs_per_print
    @number_of_epochs_per_print.setter
    def number_of_epochs_per_print(self, num):
        self._number_of_epochs_per_print=num
    def on_epoch_end(self, epoch, logs=None, ):
        if epoch % self.number_of_epochs_per_print == 0:
            print(f"Epoch {epoch}/{self.params['epochs']} - training loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")
            import os
            

class RegressionModel(IRegression):
    """Class for training and plotting history of a regression model using MLP."""
    
    @property
    def epoch_number(self):
        return self._epoch_number
    
    @epoch_number.setter
    def epoch_number(self, num):
        if self.model == None:
            self._epoch_number = num 
        else: 
            print("Cannot change the parameters of an already defined method") 

    @property
    def stopping_patience(self):
        return self._stopping_patience
    
    @stopping_patience.setter
    def stopping_patience(self, num):
        if self.model == None:
            self._stopping_patience = num
        else: print("Cannot change the parameters of an already defined method")
        

    def __init__(self, hidden_layers=1, nodes_per_layer=5, epochs=100, patience=100):
        """
        Constructor for RegressionModel_mlp class.

        Parameters:
        hidden_layers (int): Number of hidden layers.
        nodes_per_layer (int): Number of nodes in each hidden layer.
        epochs (int): Number of epochs for training.
        patience (int): Patience for early stopping.
        """
        self._hidden_layers = hidden_layers
        self._stopping_patience = patience
        self._nodes_per_layer = nodes_per_layer
        self._model = None  # Placeholder for the Keras model
        self._history = None  # Placeholder for the training history
        self._epoch_number = epochs        

    def Compile_Model(self,shape=None, checkpoint_path = None, number_of_epochs_per_print = 3):
        """Define the architecture of the regression model."""
        if (shape == None and checkpoint_path == None):
            print("Required arguments not provided.")
            return
        input_layer = Input(shape=(shape,))
        hidden_layer = input_layer
        for _ in range(self._hidden_layers):
            hidden_layer = Dense(self._nodes_per_layer, activation='relu')(hidden_layer)
        output_layer = Dense(1, activation='linear')(hidden_layer)
        self._model = Model(inputs=input_layer, outputs=output_layer)
        if(checkpoint_path != None): 
            self._model.load_weights(checkpoint_path)
        # Define callbacks
        self.checkpoint = ModelCheckpoint("checkpoints/model_checkpoint" 
                                          + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                                          + ".weights.h5"
                                          , monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=self._stopping_patience, verbose=2
                                            , mode='min', restore_best_weights=True)
        self.print_progress = PrintProgress()
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.3 , patience=50, mode = 'min'
                                           , min_lr=0.01, verbose = 1, min_delta=1)

        self.print_progress.number_of_epochs_per_print = 100        
        self.plotPath = os.path.sep.join(["output", "model.png"])
        self.jsonPath = os.path.sep.join(["output", "model.json"])
        #Compile model
        self._model.compile( loss = "mean_squared_error", optimizer='adam') 


    def Start_Training(self, X, y, save_checkpoint_path = None
                       , validation_split = 0.75, epochs = None, verbose_training = 0):
        """
        Train the regression model.

        Parameters:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        
        Returns:
        history: Training history.
        """
        if epochs == None:
            epochs = self._epoch_number
        if self._model is None:
            raise ValueError("Model has not been initialized. Please initialize the model before training.")
        if save_checkpoint_path is not None:
            self.checkpoint = ModelCheckpoint(save_checkpoint_path,  monitor='val_loss'
                                              , verbose=0, save_best_only=True, mode='min', save_weights_only=True)
        # Train the model with callbacks
        history = self._model.fit(X, y, validation_split = validation_split, batch_size = 100
                                  , epochs = epochs, verbose = verbose_training, shuffle = True
                                  , callbacks=[self.checkpoint, self.early_stopping, self.print_progress, self.reduce_lr])
        self._history = history 
        return history
    
    def Plot_History(self):
        """
        Plot the training and validation loss history.
        """
        if self._history is None:
            print("No training history available. Please train the model first.")
            return
        print(self._history.history.keys())
        # Plot training and validation loss
        plt.plot(self._history.history['loss'], label='training loss')
        plt.plot(self._history.history['val_loss'], label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def Save_Model(self, path = ""):
        if path == "":
            path= "saves/" 
        path += str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        self._model.save(path + ".keras")
        self._model.save_weights(path + ".weights.h5")


