
class IData:
    """Interface for data classes."""
    @property
    def data_grid(self):
        """Abstract property representing the data grid."""
        raise NotImplementedError("Subclasses must implement data_grid", 
                                  "property")

    @property
    def selected_column(self):
        """Abstract property representing the selected column."""
        raise NotImplementedError("Subclasses must implement selected_column",
                                  "property")


