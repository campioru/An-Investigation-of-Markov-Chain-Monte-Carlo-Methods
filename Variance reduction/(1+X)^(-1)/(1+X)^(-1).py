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

dims = [1, 10]
analytic_ints = [np.log(2.), 73.*np.pi**10. / 6842880.]

for (dim, analytic_int) in zip(dims, analytic_ints):
    base_vecs = np.random.uniform(0, 1, (iterations, dim))
    base_prods = np.prod(base_vecs, axis=-1)
    base_obs = 1. / (1. + base_prods)
    base_means = np.cumsum(base_obs) / iteration_range
    base_vars = np.array([np.inf if i == 1 else np.var(base_obs[:i], ddof=1, mean=base_means[i-1]) for i in iteration_range])
    base_error = (base_vars[-1] / iterations) ** .5
    print(f"Original variance = {base_vars[-1]}")
    print(f"Original percentage error = {100 * (base_means[-1] - analytic_int) / analytic_int}%")
    print()

    double_obs = np.empty(2 * iterations, dtype=base_obs.dtype)
    double_obs[::2] = base_obs.copy()
    double_obs[1::2] = 1. / (1. + np.prod(np.random.uniform(0, 1, (iterations, dim)), axis=-1))
    double_means = np.cumsum(double_obs) / double_point_range
    double_vars = np.array([np.inf if p == 1 else np.var(double_obs[:p], ddof=1, mean=double_means[p-1]) for p in double_point_range])
    double_errors = np.sqrt(double_vars / double_point_range)
    print(f"Double sample variance = {double_vars[-1]}")
    print(f"Double sample percentage error = {100 * (double_means[-1] - analytic_int) / analytic_int}%")
    print()

    pair_prods = np.prod(1 - base_vecs, axis=-1)
    pair_obs = 1. / (1. + pair_prods)
    pair_means = np.cumsum(pair_obs) / iteration_range
    pair_vars = np.array([np.inf if i == 1 else np.var(pair_obs[:i], ddof=1, mean=pair_means[i-1]) for i in iteration_range])
    base_pair_means, base_pair_vars = np.array([base_means, pair_means]), np.array([base_vars, pair_vars])
    anti_means = np.mean(base_pair_means, axis=0)
    anti_covs = np.array([np.inf if i == 1 else np.cov([base_obs[:i], pair_obs[:i]], ddof=1)[0, 1] for i in iteration_range])
    anti_vars = (np.mean(base_pair_vars, axis=0) + anti_covs) / 2.
    anti_errors = np.sqrt(anti_vars / iteration_range)
    print(f"Antithetic variable variance = {anti_vars[-1]}")
    print(f"Antithetic variable percentage error = {100 * (anti_means[-1] - analytic_int) / analytic_int}%")
    print()

    # TODO consider generalising to multidimensional antithetic variables
    # i.e. use all possible combinations of U_i and 1 - U_i

    variate_obs = np.array([1 + base_prods, 1 + pair_prods])
    variate_mean = 1. + .5 ** dim
    variate_means = np.cumsum(variate_obs, axis=-1) / iteration_range[None]
    variate_vars = np.array([[np.inf if i == 1 else np.var(obs[:i], ddof=1, mean=means[i-1]) for i in iteration_range] for (obs, means) in zip(variate_obs, variate_means)])
    control_covs = 1. - base_pair_means * variate_means # == sample covariance
    cstars = -control_covs / variate_vars
    control_means = base_pair_means + cstars * (variate_means - variate_mean)
    # control_covs * cstars == -control_covs**2. / variate_vars
    control_vars = base_pair_vars + control_covs * cstars
    control_errors = np.sqrt(control_vars[0] / iteration_range)
    print(f"Control variate variance = {control_vars[0, -1]}")
    print(f"Control variate percentage error = {100 * (control_means[0, -1] - analytic_int) / analytic_int}%")
    print()
    anti_control_means = np.mean(control_means, axis=0)
    base_pair_obs = np.array([base_obs, pair_obs])
    anti_control_covs = np.array([np.inf if i == 1 else np.cov(base_pair_obs[:, :i] + cstars[:, i-1, None] * (variate_obs[:, :i] - variate_mean), ddof=1)[0, 1] for i in iteration_range])
    anti_control_vars = (np.mean(control_vars, axis=0) + anti_control_covs) / 2.
    anti_control_errors = np.sqrt(anti_control_vars / iteration_range)
    print(f"Control variate + antithetic variable variance = {anti_control_vars[-1]}")
    print(f"Control variate + antithetic variable percentage error = {100 * (anti_control_means[-1] - analytic_int) / analytic_int}%")
    print()

    # TODO try jackknifing for control(antithetic)
    # variate_obs = 2. / (base_obs + pair_obs)
    # control_anti_obs(j) = anti_obs(j) + cstar_jack(-j) * (variate_obs(j) - variate_means(-j))
    # control_anti_var = something complicated
    # TODO then generalise to K-fold cross-fitting

    plt.axhline(y=analytic_int, color="k", linestyle="dashed")
    plt.plot(double_point_range / 2, double_means, color="r", label="Double")
    plt.fill_between(double_point_range / 2, double_means - double_errors, double_means + double_errors, color="r", alpha=.25)
    plt.plot(iteration_range, anti_means, color="g", label="Antithetic")
    plt.fill_between(iteration_range, anti_means - anti_errors, anti_means + anti_errors, color="g", alpha=.25)
    plt.plot(iteration_range, control_means[0], color="b", label="Control")
    plt.fill_between(iteration_range, control_means[0] - control_errors, control_means[0] + control_errors, color="b", alpha=.25)
    plt.plot(iteration_range, anti_control_means, color="purple", label="Control + antithetic")
    plt.fill_between(iteration_range, anti_control_means - anti_control_errors, anti_control_means + anti_control_errors, color="purple", alpha=.25)
    plt.xscale("log", base=2)
    plt.xlim(1, iterations)
    plt.xlabel("iterations")
    plt.ylabel(r"$\bar{f}$")
    plt.title(r"$\int_0^1 \frac{1}{1+X^{(" + f"{dim}" + r")}}\,dx$")
    plt.legend()
    plt.savefig(f"(1+X^({dim}))^(-1).pdf", bbox_inches='tight')
    plt.clf()
