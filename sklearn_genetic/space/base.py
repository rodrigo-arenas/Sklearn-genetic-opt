from abc import ABC, abstractmethod
from typing import Any


class BaseDimension(ABC):
    """
    Base class for the space definition of data types
    """

    @abstractmethod
    def sample(self) -> Any:
        """
        Sample a random value from the assigned distribution
        """

        raise NotImplementedError(
            "The sample method must be defined according each data type handler"
        )  # pragma: no cover
