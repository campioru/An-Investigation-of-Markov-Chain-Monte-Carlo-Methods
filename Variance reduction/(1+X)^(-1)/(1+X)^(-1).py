"""An Investigation of Markov Chain Monte Carlo Methods: (1+X)^(-1).

Employs the antithetic variable and control variate approaches to reduce the
variance of the estimate of the integral of (1+X)^(-1) from 0 to 1, with X = X1
and X = X1...X10.

@author: Ruaidhrí Campion
"""

import numpy as np
import matplotlib.pyplot as plt


K = 20
n = 2 ** K
qs = [1, 10]
analytic_means = [np.log(2.), 73*np.pi**10. / 6842880.]
for (q, analytic_mean) in zip(qs, analytic_means):
    xs = np.random.uniform(0, 1, (n, q))
    xs_ = xs.copy()
    xs_[1::2] = 1. - xs[::2]
    Xs = np.prod(xs, axis=-1)
    Xs_ = np.prod(xs_, axis=-1)
    d_s = 1. / (1. + Xs)
    a_s = 1. / (1. + Xs_)

    mu = 1. + .5 ** q
    var = 3.**-q - .5**(2*q)
    # TODO change to sample covariance at to each point
    # otherwise comparing method using all the data to methods involving a subset of the data
    cov = np.mean((1. + Xs) * d_s) - mu * np.mean(d_s)
    cstar = -cov / var
    c_s = d_s + cstar * ((1. + Xs) - mu)

    points = np.arange(2, n+1, 2)
    average_ds = np.cumsum(d_s)[1::2] / points
    average_as = np.cumsum(a_s)[1::2] / points
    average_cs = np.cumsum(c_s)[1::2] / points

    # TODO include errorbars (include autocorrelation)
    plt.axhline(y=analytic_mean, color="k", linestyle="dashed")
    plt.plot(points, average_ds, color="r", label="Direct")
    plt.plot(points, average_as, color="g", label="Antithetic")
    plt.plot(points, average_cs, color="b", label="Control")
    plt.xscale("log", base=2)
    plt.xlim(2, n)
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\bar{f}$")
    plt.title(r"$\int_0^1 \frac{1}{1+X^(" + f"{q}" + r")}\,dx$")
    plt.legend()
    plt.savefig(f"(1+X^({q}))^(-1).pdf", bbox_inches='tight')
    plt.clf()

# TODO consider combining antithetic + control
