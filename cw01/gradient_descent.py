from dataclasses import dataclass
from typing import Callable, List
from math import sqrt
from copy import deepcopy


@dataclass
class SolverConfig:
    f: Callable[[List[float]], float]
    x0: List[float]
    alpha: float
    eps: float
    domain: float = 100
    max_it: int = int(1e3)


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


def solver(config: SolverConfig):
    f, x0, alpha, eps, domain, max_it = (
        config.f,
        config.x0,
        config.alpha,
        config.eps,
        config.domain,
        config.max_it,
    )

    values = []
    pos_change, grad_norm = eps, eps

    while pos_change >= eps and grad_norm >= eps and max_it:
        max_it -= 1
        f_grad = gradient(f, x0)
        grad_norm = sum(grad**2 for grad in f_grad)
        x1 = [x - alpha * grad for x, grad in zip(x0, f_grad)]
        pos_change = sqrt(sum([(a - b) ** 2 for a, b in zip(x1, x0)]))
        values.append(f(x0))
        x0 = x1
        for i in range(len(x0)):
            if x0[i] > domain:
                x0[i] = domain
            if x0[i] < -domain:
                x0[i] = -domain
    values.append(f(x0))
    return x0, values
