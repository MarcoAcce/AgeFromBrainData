from keras.callbacks import Callback

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
            print(f"Epoch {epoch}/{self.params['epochs']} -" ,
                  f"training loss: {logs['loss']:.4f} -" ,
                  f"val_loss: {logs['val_loss']:.4f}")


            

