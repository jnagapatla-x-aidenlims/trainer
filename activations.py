import math

def tanh(value):
    return math.tanh(value)

def tanh_prime(value):
    """
    Returns the derivative of the hyperbolic tangent

    f'(x) = 1 - f^2(x)
    """
    return 1 - math.tanh(value) ** 2
