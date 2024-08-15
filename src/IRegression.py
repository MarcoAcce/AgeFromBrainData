class IRegression:
    """Interface for regression classes."""
    
    def Compile_Model(self, shape=None, checkpoint_path = None, 
                      number_of_epochs_per_print = 0):
        """Abstract method for defining the model.
        
        Parameters:
        
        shape: the shape of the data for training and evaluation.
        
        checkpoint_path (string): Path to the checkpoint to be loaded. 
        If left empty no checkpoint will be used.
        
        number_of_epochs_per_print (int): Number of epochs between prints of 
        the PrintProgress callback during training.
        """
        raise NotImplementedError("Subclasses must implement Compile_Model" ,
                                  "method")

    def Start_Training(self, X, y, save_checkpoint_path = None, 
                       validation_split = 0.75, epochs = None, 
                       verbose_training = 0):
        """
        Abstract method for starting the training.
        
        Parameters:
        
        X (numpy.ndarray): Input features.
        
        y (numpy.ndarray): Target values.
        
        save_checkpoint_path (string): Path to the directory used for saving 
        checkpoints during training.
        
        validation_split (float): Fraction used to define the 
        training/validation split for the model. Default at 0.75.
        
        epochs (int): Number of epochs for training. Default at the value set 
        in the constructor.
        
        Returns:
        
        history: Training history.
        """
        raise NotImplementedError("Subclasses must implement Start_Training",
                                  "method")

    def Plot_History(self):
        """Abstract method for plotting the training history."""
        raise NotImplementedError("Subclasses must implement Plot_History",
                                  "method")