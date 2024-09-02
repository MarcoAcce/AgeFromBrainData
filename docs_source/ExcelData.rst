Excel data class
================

Allows to extract a Pandas dataframe from a .xlsx file and perform basic manipulations to create numpy arrays suitable for machine learning.
The data in the data frame is extracted into numpy arrays for model training and inference.
The initial data frame will be preserved as is in the instance of the class, all manipulations will act directly on the numpy arrays read from the data frame.

.. automodule:: ExcelData
   :special-members: __init__
   :members:
   :show-inheritance:
