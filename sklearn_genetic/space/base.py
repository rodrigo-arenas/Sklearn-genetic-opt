class BaseDimension(object):
    """
    Base class for the space definition of data types
    """

    def sample(self):
        """
        Sample a random value from the assigned distribution
        """

        raise NotImplementedError(
            "The sample method must be defined according each data type handler"
        )
