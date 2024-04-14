import pandas
import numpy
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input

from Data import IData,ExcelData

#input data from .xlsx files

samples = ExcelData('input\FS_features_ABIDE_males.xlsx')

my_matrix = samples.data_grid
age = samples.selected_column('AGE_AT_SCAN')

input_layer = Input( shape= (len(my_matrix[0]), ) )
hidden_1 = Dense(5,activation='relu')(input_layer)
#hidden_2 = Dense(100,activation='relu')(hidden_1)
output = Dense(1,activation='relu')(hidden_1)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam')

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
class PrintProgress(Callback): #defined to have a printout of the training process, not mandatory
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            print(f"Epoch {epoch}/{self.params['epochs']} - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")

# Define callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', verbose=0, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='min', restore_best_weights=True)
print_progress = PrintProgress()
# Include callbacks from MLP model
my_callbacks = [checkpoint, print_progress]

# Train the model with callbacks
history = model.fit(my_matrix, age, validation_split=0.25, epochs=30, verbose=0, callbacks=my_callbacks)


print(history.history.keys())# Print keys available in the history
# Plot training and validation loss
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()