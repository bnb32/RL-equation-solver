"""Custom operator functions"""
from operator import pow


def div(a, b):
    """Define divide operator"""
    return a / b


def root(a, b):
    """Define root function"""
    return pow(a, 1 / b)


# pylint: disable=unused-argument
def sqrt(a, b):
    """Define sqrt function"""
    return a ** 0.5


# pylint: disable=unused-argument
def square(a, b):
    """Define square function"""
    return a * a
