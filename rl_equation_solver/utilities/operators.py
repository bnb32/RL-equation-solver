"""Custom operator functions"""
from operator import pow


def div(a, b):
    """Define divide operator"""
    return a / b


def root(a, b):
    """Define root function"""
    return pow(a, pow(b, -1))


# pylint: disable=unused-argument
def sqrt(a, b):
    """Define sqrt function"""
    return a ** 0.5


# pylint: disable=unused-argument
def square(a, b):
    """Define square function"""
    return a * a


def fraction(a):
    """Return float from fraction"""
    if "/" in a:
        a, b = a.split("/")
        return int(a) / int(b)
    else:
        return float(a)
