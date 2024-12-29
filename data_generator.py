import numpy as np
from itertools import product


class DataGenerator:
    def __init__(self):
        pass

    def get_data(self, domain, dim, fun, samples_density):
        domain_min = domain[0]
        domain_max = domain[1]
        x = np.array([np.linspace(domain_min, domain_max, samples_density).reshape(-1, 1) for _ in range(dim)])
        x = np.linspace(domain_min, domain_max, samples_density)
        X = np.array(list(product(x, repeat=dim)))
        Y = fun(X)
        # for _ in range(dim-1):
        #     X = np.array(list(product(X, x)))
        return X, Y