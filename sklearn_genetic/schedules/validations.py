from .base import BaseAdapter
from .schedulers import ConstantAdapter


def check_adapter(adapter):
    """
    Check if the adapter is a number of a class, if it's a number it will create a ConstantAdapter
    """
    if isinstance(adapter, BaseAdapter):
        return adapter
    elif (isinstance(adapter, int) or isinstance(adapter, float)) and not isinstance(adapter, bool):
        return ConstantAdapter(initial_value=adapter, end_value=adapter, adaptive_rate=0)
    else:
        raise ValueError(
            "adapter should be either a class with inheritance from schedulers.base.BaseAdapter "
            "or a real number."
        )
