from copy import deepcopy
from math import sqrt


def gradient(f, x0):
    h = 1e-8
    dim = len(x0)
    f_value = f(x0)
    grad = []
    for i in range(dim):
        x0_h = deepcopy(x0)
        x0_h[i] += h
        f_h_value = f(x0_h)
        f_prim = (f_h_value - f_value) / h
        grad.append(f_prim)
    return grad


def solver(f, x0, alpha, eps, domain=100, max_it=1e3):
    values = [f(x0)]
    pos_change, grad_norm = eps, eps
    while max_it:
        max_it -= 1
        f_grad = gradient(f, x0)
        grad_norm = sum(grad**2 for grad in f_grad)
        x1 = [x - alpha * grad for x, grad in zip(x0, f_grad)]
        pos_change = sqrt(sum([(a - b) ** 2 for a, b in zip(x1, x0)]))
        values.append(f(x0))
        x0 = x1
        for x in x0:
            if x > domain:
                x = domain
            if x < -domain:
                x = -domain
    values.append(f(x1))
    return x1, values
