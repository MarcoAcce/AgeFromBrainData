import os
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.callbacks import  ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from datetime import datetime

from IRegression import IRegression
from Callbacks import PrintProgress

class RegressionModel(IRegression):
    """
    Class for training and plotting history of a regression model using MLP.
    """
    def __init__(self, 
                 saved_model_path: str = None ):
        """
        Constructor for RegressionModel class.
        Can be used to either define a new model for training or for 
        loading an already trained model.

        Parameters:
        
        saved_model_path (string): Path to a saved model, should
        point to a .keras file.
        """
        self._history = None  # Placeholder for the training history
        if saved_model_path != None: 
            try: self._model = self._load_model(saved_model_path)
            except Exception: print("Error while loading the model!")
        else: self._model = None  # Placeholder for the Keras model

       

    def Compile_Model(self,
                      shape, 
                      hidden_layers: int = None, 
                      nodes_per_layer: int = None):
        """
        Define the architecture of the regression model.
        Shouldn't be used with models loaded from .keras files.
        
        Parameters:
        
        shape: the shape of the data for training and evaluation.
        
                
        hidden_layers (int): Number of hidden layers.
        
        nodes_per_layer (int): Number of nodes in each hidden layer.
        
        verbose_training (int): Determines the frequency of prints during
        traing. 0 No prints, 1 prints every 100 epochs, 
        2 prints every 10 epochs, 3 prints for every epoch. Default is 1.
        """
        self._hidden_layers = hidden_layers
        self._nodes_per_layer = nodes_per_layer
        input_layer = Input(shape=(shape,))
        hidden_layer = input_layer
        
        for _ in range(self._hidden_layers):
            hidden_layer = Dense(self._nodes_per_layer, 
                                 activation='relu')(hidden_layer)
        output_layer = Dense(1, activation='linear')(hidden_layer)
        self._model = Model(inputs=input_layer, outputs=output_layer)
        self._model.compile( loss = "mean_squared_error", optimizer='adam') 


    def Start_Training(self, 
                       X, 
                       y, 
                       save_checkpoint_path: str = None, 
                       validation_split: float = 0.75, 
                       epochs: int = 10000, 
                       stopping_patience: int = 1000, 
                       verbose_training: int = 1):
        """
        Train the regression model.

        Parameters:
        
        X (numpy.ndarray): Input features.
        
        y (numpy.ndarray): Target values.
        
        save_checkpoint_path (string): Path to the directory used for saving
        checkpoints during training.
        
        validation_split (float): Fraction used to define the 
        training/validation split for the model. Default at 0.75.
        
        epochs (int): Number of epochs for training. Default value 
        is 10000.
        
        stopping_patience (int): Number of epoch used for the Early stopping
        callback. Default value is 1000.
        
        verbose_training (int): Determines the frequency of prints during
        training. 0 No prints, 1 prints every 100 epochs, 
        2 prints every 10 epochs, 3 prints for every epoch. Default is 1.
        
        Returns:
        
        history: Training history.
        """       
        if self._model is None:
            raise ValueError("Model has not been initialized.",
                             "Must initialize the model before training.")
        
        # Define callbacks    
        self._define_callbacks(stopping_patience, verbose_training, 
                               save_checkpoint_path)
        
        # Train the model with callbacks
        history = self._model.fit(X, y, validation_split = validation_split,
                                  batch_size = 100, epochs = epochs, 
                                  verbose = 0, shuffle = True, 
                                  callbacks = self._callbacks)
        
        self._history = history 
        
        training_result = self._model.evaluate(X,y, verbose = 0)
        print("Loss and accuracy over training and validation data:", 
              training_result)
        
        return history
    
    def Plot_History(self):
        """
        Plot the training and validation loss history. 
        Saves the plot as a .png.
        """
        
        if self._history is None:
            print("""No training history available. 
                  Please train the model first.""")
            return
        
        print(self._history.history.keys())
        
        # Plot training and validation loss
        plt.plot(self._history.history['loss'], label='training loss')
        plt.plot(self._history.history['val_loss'], label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(left=100)
        plt.legend()
        path = self._save_path + '.png'
        plt.savefig(path)
        plt.show()

    def Save_Model(self, path: int = ""):
        path = self._save_path
        self._model.save(path + ".keras")
        self._model.save_weights(path + ".weights.h5")

    
    def Predict(self, X):
        return self._model.predict(X)
    
                
    @property
    def _save_path(self):
        """
        Property containg the general name for all saved files.
        Will create a save directory the first time its called.
        """
        save_path = os.path.join(os.getcwd(), r'saves')
        os.makedirs(save_path, exist_ok = True)
        save_path += "\\" + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        return save_path    

    
    def _define_callbacks(self, 
                          stopping_patience, 
                          verbose_training, 
                          save_checkpoint_path):
        """
        Defines all callbacks for training.
        """
        self._callbacks = []
        
        early_stopping = EarlyStopping(monitor='val_loss', 
                                           patience=stopping_patience,
                                           verbose=2, mode='min',
                                           restore_best_weights=True)
        self._callbacks.append(early_stopping)
        
        if verbose_training != 0:
            print_progress = PrintProgress()
            if verbose_training == 3:
                print_progress.number_of_epochs_per_print = 1
            if verbose_training == 2:
                print_progress.number_of_epochs_per_print = 10
            else:
                print_progress.number_of_epochs_per_print = 100
                if verbose_training != 1:
                    print("Verbose_traning argument not valid",
                          ",defaulting to 1.")
            self._callbacks.append(print_progress)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.3, 
                                          patience=50, mode = 'min',
                                          min_lr=0.01, verbose = 1,
                                          min_delta=1)
        self._callbacks.append(reduce_lr)

        if save_checkpoint_path is None: 
            save_checkpoint_path = ("checkpoints/model_checkpoint" + 
                                   str(datetime.now().strftime(
                                       "%Y-%m-%d %H-%M-%S") ) + 
                                   ".weights.h5")
        
        checkpoint = ModelCheckpoint(save_checkpoint_path,
                                             monitor='val_loss', verbose=0,
                                             save_best_only = True, 
                                             mode='min', 
                                             save_weights_only = True)
        self._callbacks.append(checkpoint)
    
    def _load_model(self,
                    saved_model_path):
        """
        Constructor for RegressionModel class, to be used to load
        an already trained model.
        This will load and uncompiled model, usable for inference but
        not for training.
        
        Parameters:
      
        saved_model_path (string): Path to a saved model, should
        point to a .keras file.
        """
        if not saved_model_path.endswith('.keras'):
            raise ValueError("Path doesn't point to a .keras file.")
        self._model = load_model(saved_model_path, compile=False)