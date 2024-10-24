from cec2017.simple import f3
from cec2017.hybrid import f12


def quadratic_fun(params):
    return sum(x**2 for x in params)


def my_f3(params):
    return f3([params])[0]


def my_f12(params):
    return f12([params])[0]
