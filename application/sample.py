"""
This is a docstring for the module:
Testing: https://www.coursera.org/learn/principles-of-computing-1/supplement/bUWIL/building-tests-for-python-programs
This module demonstrates the use of docstrings.
"""

def repeat_string(string, num):
    """
    This is a docstring for a function:

    Return a new string that repeats the input
    string num times.
    """
    newstring = ""
    for dummy_loop_counter in range(num):
        newstring += string
    return newstring


class SimpleCounter:
    """
    This is a docstring for a class:

    Simple counter class that can only increment.
    """

    def __init__(self, initial_val=0):
        self._val = initial_val

    def increment(self):
        """
        This is a docstring for a method:

        Increment counter.
        """
        self._val += 1

    def get_value(self):
        """
        This is a docstring for another method:

        Return current value of the counter.
        """
        return self._val

counter=SimpleCounter(5)
counter.increment()
counter.increment()
print(counter.get_value())