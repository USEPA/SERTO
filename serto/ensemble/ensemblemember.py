# python imports

# third-party imports

# local imports

class SWMMEnsembleMember(object):
    """
    This class represents an ensemble member
    # after base file copy
    # after initialize callbacks
    # after step callbacks
    # after end callbacks
    # after finalize callbacks
    """

    def __init__(self, model, data):
        """
        Constructor for the ensemble member
        :param model: The model for the ensemble member
        :param data: The data for the ensemble member
        """
        self.model = model
        self.data = data

    def __str__(self):
        return f"EnsembleMember(model={self.model}, data={self.data})"

    def __repr__(self):
        return self.__str__()
