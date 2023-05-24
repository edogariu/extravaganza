import jax
import jax.numpy as jnp

# Convenience to work with both normal and mutable numbers
def _v(obj):
    """
    gets value of object
    """
    if isinstance(obj, FloatWrapper):
        return obj.get_value()
    else:
        return obj

class FloatWrapper:
    """
    TODO 
    eventually implement all the things you get when you call `print((0.).__dir__())`
    """

    def __init__(self, value: float):
        """
        mutable float wrapper!

        Parameters
        ----------
        value : float
            float's value
        """
        if isinstance(value, jax.Array):
            self.value = value.clone()
        elif isinstance(value, float):
            self.value = jnp.array([value])
        else:
            raise ValueError(value)
        print('hi', self.value)
    
    def get_value(self) -> float:
        print(self.value)
        return float(self.value.item())
    
    def set_value(self, value: float):
        if isinstance(value, jax.Array):
            self.value = value.clone()
        else:
            self.value = jnp.array([value])
            
    # -----------------------------------------------------------------------------------------------
    # ------------------------------------ EVERYTHING ELSE ------------------------------------------
    # -----------------------------------------------------------------------------------------------

    # comparison interface
    def __eq__(self, _value: float) -> bool:
        return _v(self) == _v(_value)

    def __ne__(self, _value: float) -> bool:
        return _v(self) != _v(_value)
    
    def __bool__(self) -> bool:
        return _v(self) != 0

    # math interface
    def __add__(self, __value: float) -> float:
        return FloatWrapper(_v(self) + _v(__value))

    def __mul__(self, __value: float) -> float:
        return FloatWrapper(_v(self) * _v(__value))
    
    def __sub__(self, __value: float) -> float:
        return FloatWrapper(_v(self) - _v(__value))
    
    def __radd__(self, __value: float) -> float:
        return self.__add__(__value)
    
    def __rmul__(self, __value: float) -> float:
        return self.__mul__(__value)
    
    def __rsub__(self, __value: float) -> float:
        return FloatWrapper(_v(__value) - _v(self))
    
    def __abs__(self) -> float:
        return FloatWrapper(abs(_v(self)))
    
    def __floor__(self) -> int:
        return FloatWrapper(_v(self).__floor__())
    
    # In-place operations alter the shared location
    def __iadd__(self, other):
        self.value += _v(other)
        return self

    def __imul__(self, other):
        self.value *= _v(other)
        return self

    # Define the copy interface
    def __copy__(self):
        new = FloatWrapper(0)
        new.value = self.value
        return new

    def __repr__(self):
        return repr(self.value.item())
    
    
if __name__ == '__main__':
    import math
    import numpy as np
    v = FloatWrapper(4.5)
    print(v)
    print(np.sqrt(v))
    q = v
    q.set_value(9.)
    print(3-v)
    q.set_value(-3)
    
    l = 30 + v
    print(l.__class__)