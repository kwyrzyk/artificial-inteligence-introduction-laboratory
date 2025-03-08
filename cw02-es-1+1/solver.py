from functions import quadratic, my_f3 as f3, my_f7 as f7
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

@dataclass
class params:
    sig: float


def solver(fun, x, sig, mut_freq, t_max, mul_up=1.22, mul_down=0.82):
    x = np.array(x)
    dim = x.shape
    t = 1
    success_num = 0
    min_val = fun(x)
    values = [min_val]
    while t <= t_max:
        new_x = x + sig * np.random.normal(0, 1, dim)
        new_x_val = fun(new_x)
        if new_x_val < min_val:
            x = new_x
            min_val = new_x_val
            success_num += 1
        values.append(min_val)
        if t % mut_freq == 0:
            if success_num / mut_freq > 0.2:
                sig *= mul_up
            else:
                sig *= mul_down
            success_num = 0
        t += 1
    return x, min_val, values
