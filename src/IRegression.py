class IRegression:
    """Interface for regression classes."""
    
    def Compile_Model(self,shape, hidden_layers: int, 
                      nodes_per_layer: int):
        """Abstract method for defining the model.
        
        Parameters:
        
        shape: the shape of the data for training and evaluation.
        
                
        hidden_layers (int): Number of hidden layers.
        
        nodes_per_layer (int): Number of nodes in each hidden layer.
        
        verbose_training (int): Determines the frequency of prints during
        traing. 0 No prints, 1 prints every 100 epochs, 
        2 prints every 10 epochs, 3 prints for every epoch. Default is 1.
        """
        raise NotImplementedError("Subclasses must implement Compile_Model" ,
                                  "method")

    def Start_Training(self, X, y, save_checkpoint_path: str, 
                       validation_split: float, epochs: int, 
                       stopping_patience: int, 
                       verbose_training: int):
        """
        Abstract method for starting the training.
        
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
        raise NotImplementedError("Subclasses must implement Start_Training",
                                  "method")

    def Plot_History(self):
        """Abstract method for plotting the training history."""
        raise NotImplementedError("Subclasses must implement Plot_History",
                                  "method")