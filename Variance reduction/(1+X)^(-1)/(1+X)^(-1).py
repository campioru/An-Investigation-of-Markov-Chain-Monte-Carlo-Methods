"""An Investigation of Markov Chain Monte Carlo Methods: (1+X)^(-1).

Employs the antithetic variable and control variate approaches to reduce the
variance of the estimate of the integral of (1+X)^(-1) from 0 to 1, with X = X1
and X = X1...X10, compared to doubling the number of samples.

@author: Ruaidhrí Campion
"""

import numpy as np
import matplotlib.pyplot as plt
from sys import argv


iterations = int(argv[1])
iteration_range = 1 + np.arange(iterations)
double_point_range = 1 + np.arange(2 * iterations)
anti_point_range = double_point_range[1::2]

dims = [1, 10]
analytic_ints = [np.log(2.), 73.*np.pi**10. / 6842880.]

for (dim, analytic_int) in zip(dims, analytic_ints):
    base_vecs = np.random.uniform(0, 1, (iterations, dim))
    base_prods = np.prod(base_vecs, axis=-1)
    base_obs = 1. / (1. + base_prods)
    base_means = np.cumsum(base_obs) / iteration_range

    double_obs = np.empty(2 * iterations, dtype=base_obs.dtype)
    double_obs[::2] = base_obs.copy()
    double_obs[1::2] = 1. / (1. + np.prod(np.random.uniform(0, 1, (iterations, dim)), axis=-1))
    double_means = np.cumsum(double_obs) / double_point_range

    anti_obs = 1. / (1. + np.prod(1 - base_vecs, axis=-1))
    anti_means = np.cumsum(base_obs + anti_obs) / anti_point_range

    variate_obs = 1 + base_prods
    variate_mean = 1. + .5 ** dim
    variate_means = np.cumsum(variate_obs) / iteration_range
    variate_vars = np.array([np.inf if i == 1 else np.var(variate_obs[:i], ddof=1) for i in iteration_range])
    control_covs = 1. - base_means * variate_means # == sample covariance of base and variate
    control_means = base_means - control_covs / variate_vars * (variate_means - variate_mean)

    plt.axhline(y=analytic_int, color="k", linestyle="dashed")
    plt.plot(double_point_range / 2, double_means, color="r", label="Double")
    plt.plot(iteration_range, anti_means, color="g", label="Antithetic")
    plt.plot(iteration_range, control_means, color="b", label="Control")
    plt.xscale("log", base=2)
    plt.xlim(1, iterations)
    plt.xlabel("iterations")
    plt.ylabel(r"$\bar{f}$")
    plt.title(r"$\int_0^1 \frac{1}{1+X^{(" + f"{dim}" + r")}}\,dx$")
    plt.legend()
    plt.savefig(f"(1+X^({dim}))^(-1).pdf", bbox_inches='tight')
    plt.clf()

# TODO consider combining antithetic + control
