"""Custom operator functions."""
from operator import pow


def nth_root(n: int):
    """Define nth root operator."""

    def func(a: int, b: int) -> float:
        return pow(a, 1 / n)

    return func


def div(a: float, b: float) -> float:
    """Define divide operator."""
    return a / b


def root(a: float, b: float) -> float:
    """Define root function."""
    return pow(a, pow(b, -1))


# pylint: disable=unused-argument
def sqrt(a: float, b: float) -> float:
    """Define sqrt function."""
    return a**0.5


# pylint: disable=unused-argument
def square(a: float, b: float) -> float:
    """Define square function."""
    return a * a


def fraction(a: str) -> float:
    """Return float from fraction."""
    if "/" in a:
        a, b = a.split("/")
        return float(a) / float(b)
    return float(a)
