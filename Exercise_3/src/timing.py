import time
from contextlib import contextmanager


@contextmanager
def timer(name="Operation"):
    """
    Context manager for timing operations

    Usage:
        with timer("Training") as t:
            # training code
        print(f"Training took {t.elapsed:.2f} seconds")
    """

    class Timer:
        def __init__(self):
            self.elapsed = 0

    t = Timer()
    start = time.time()
    yield t
    t.elapsed = time.time() - start
    print(f"{name} took {t.elapsed:.2f} seconds")


def time_function(func):
    """
    Decorator for timing functions

    Usage:
        @time_function
        def my_function():
            # code
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result

    return wrapper