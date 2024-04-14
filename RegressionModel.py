import os
import signal
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import pickle
import datetime
from ExcelData import IData,ExcelData

class PrintProgress(Callback):
    """Callback for printing progress during model training."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            print(f"Epoch {epoch}/{self.params['epochs']} - training loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")
            import os
            

class RegressionModel:
    """Class for training and plotting history of a regression model using MLP."""
    
    @property
    def epoch_number(self):
        return self._epoch_number
    
    @epoch_number.setter
    def epoch_number(self, num):
        if self.model == None:
            self._epoch_number = num 
        else: print("Cannot change the parameters of an already defined method") 

    @property
    def stopping_patience(self):
        return self._stopping_patience
    
    @stopping_patience.setter
    def stopping_patience(self, num):
        if self.model == None:
            self._stopping_patience = num
        else: print("Cannot change the parameters of an already defined method")

    def __init__(self, hidden_layers=1, nodes_per_layer=5, epochs=100, patience=100, from_history_path=None, from_checkpoint_path=None):
        """
        Constructor for RegressionModel_mlp class.

        Parameters:
        hidden_layers (int): Number of hidden layers.
        nodes_per_layer (int): Number of nodes in each hidden layer.
        epochs (int): Number of epochs for training.
        patience (int): Patience for early stopping.
        from_history_path (str): File path to saved history.
        from_checkpoint_path (str): File path to saved model checkpoint.
        """
        self._hidden_layers = hidden_layers
        self._stopping_patience = patience
        self._nodes_per_layer = nodes_per_layer
        self._model = None  # Placeholder for the Keras model
        self._history = None  # Placeholder for the training history
        self._epoch_number = epochs
        # Define callbacks
        self.checkpoint = ModelCheckpoint("checkpoints/model_checkpoint.keras", monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=self._stopping_patience, verbose=2, mode='min', restore_best_weights=True)
        self.print_progress = PrintProgress()

        if from_history_path is not None and from_checkpoint_path is not None:
            self._load_model_from_checkpoint(from_checkpoint_path)
            self._load_history(from_history_path)
        
        # Register signal handler for keyboard interrupts
        #signal.signal(signal.SIGINT, self._handle_interrupt)

    def _load_model_from_checkpoint(self, checkpoint_path):
        """Load model from a checkpoint file."""
        if os.path.exists(checkpoint_path):
            print("Loading model from checkpoint...")
            self._model = load_model(checkpoint_path)
        else:
            print("Checkpoint file not found. Model initialization skipped.")

    def _load_history(self, history_path):
        """Load training history from a file."""
        if os.path.exists(history_path):
            print("Loading training history...")
            with open(history_path, 'rb') as file:
                self._history = pickle.load(file)
        else:
            print("History file not found. History initialization skipped.")


    def Define_Model(self, shape):
        """Define the architecture of the regression model."""
        input_layer = Input(shape=(shape,))
        hidden_layer = input_layer
        for _ in range(self._hidden_layers):
            hidden_layer = Dense(self._nodes_per_layer, activation='relu')(hidden_layer)
        output = Dense(1, activation='linear')(hidden_layer)
        self._model = Model(inputs=input_layer, outputs=output)
        self._model.compile(loss='mean_squared_error', optimizer='adam')

    def Start_Training(self, X, y, save_checkpoint_path = None, validation_split = 0.5, epochs = None, verbose_training = 0):
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
            self.checkpoint = ModelCheckpoint(save_checkpoint_path, monitor='val_loss', verbose = verbose_training, save_best_only=True, mode='min')
        # Train the model with callbacks
        history = self._model.fit(X, y, validation_split = validation_split, epochs = epochs,verbose = verbose_training, callbacks=[self.checkpoint, self.early_stopping, self.print_progress])
        self._handle_interrupt(None,None)
        self._history = history 
        return history
    
    def Resume_Training(self, X, y, from_checkpoint_path = "checkpoints/model_checkpoint.keras", save_checkpoint_path=None, epochs = None, validation_split = 0.5, verbose_training = 0):
        """
        Resume training from the last saved checkpoint.

        Parameters:
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target values.
        
        Returns:
        history: Training history.
        """
        if os.path.exists(from_checkpoint_path):
            print("Resuming training from the last checkpoint...")
            self._model.load_weights(from_checkpoint_path)
            history = self.Start_Training(X,y,save_checkpoint_path, epochs, validation_split, verbose_training)
            return history
        else:
            print("Checkpoint file not found. Cannot resume training.")
            return None


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
    
    def _handle_interrupt(self, signum, frame):
        """
        Handle keyboard interrupt signal.

        Currently doesn't work
        
        Parameters:
        signum: Signal number
        frame: Current stack frame
        """
        pass
    """
        #print("\nTraining stopped. Saving model checkpoint...")
        if self._model is not None:
            # Get the current date and time
            current_datetime = datetime.datetime.now()
            # Format the date and time as a string
            formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
            fileName = "checkpoints/interrupted_model_checkpoint-" + formatted_datetime 
            self._model.save_weights(fileName + ".weights.h5")
            #rename
            os.rename(fileName + ".weights.h5", fileName + ".keras")
        if self._history is not None:
            history_path = os.path.join(os.getcwd(), "interrupted_model_history.pkl")
            with open(history_path, "wb") as f:
                pickle.dump(self._history.history, f)
        exit(0) 
        """


