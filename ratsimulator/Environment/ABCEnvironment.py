import abc


class ABCEnvironment(metaclass=abc.ABCMeta):
    """
    Abstract/skeleton class for alternative
    environments that the animal can traverse.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def sample_uniform(self):
        """
        Sample a location in the environment uniformly

        Required for tiling the environment uniformly with
        place cells (centers)
        """
        pass

