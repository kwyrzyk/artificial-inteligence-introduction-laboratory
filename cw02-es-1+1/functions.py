from cec2017.simple import f3, f7


def quadratic(params):
    return sum([x**2 for x in params])


def my_f3(params):
    return f3([params])[0]


def my_f7(params):
    return f7([params])[0]
