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

xs = np.random.uniform(0, 1, n)
xs_ = xs.copy()
xs_[1::2] = 1. - xs[::2]
d_s = 1. / (1. + xs)
a_s = 1. / (1. + xs_)

mu = 1.5
var = 1./3. - 1./(2.**2.)
# TODO change to sample covariance at to each point
# otherwise comparing method using all the data to methods involving a subset of the data
cov = np.mean((1. + xs) * d_s) - mu * np.mean(d_s)
cstar = -cov / var
c_s = d_s + cstar * ((1. + xs) - mu)

points = np.arange(2, n+1, 2)
average_ds = np.cumsum(d_s)[1::2] / points
average_as = np.cumsum(a_s)[1::2] / points
average_cs = np.cumsum(c_s)[1::2] / points

# TODO include errorbars (include autocorrelation)
plt.axhline(y=np.log(2.), color="k", linestyle="dashed")
plt.plot(points, average_ds, color="r", label="Direct")
plt.plot(points, average_as, color="g", label="Antithetic")
plt.plot(points, average_cs, color="b", label="Control")
plt.xscale("log", base=2)
plt.xlim(2, n)
plt.xlabel(r"$n$")
plt.ylabel(r"$\bar{f}$")
plt.title(r"$\int_0^1 \frac{1}{1+x}\,dx$")
plt.legend()
plt.savefig("(1+X1)^(-1).pdf", bbox_inches='tight')
plt.clf()
# TODO consider combining antithetic + control


# TODO combine below into loop with above
K = 20
n = 2 ** K
q = 10

xs = np.random.uniform(0, 1, (n, q))
xs_ = xs.copy()
xs_[1::2] = 1. - xs[::2]
Xs = np.prod(xs, axis=-1)
Xs_ = np.prod(xs_, axis=-1)
d_s = 1. / (1. + Xs)
a_s = 1. / (1. + Xs_)

mu = 1. + .5 ** q
var = 1./(3.**q) - 1./(2.**(2.*q))
cov = np.mean((1. + Xs) * d_s) - mu * np.mean(d_s)
cstar = -cov / var
c_s = d_s + cstar * ((1. + Xs) - mu)

points = np.arange(2, n+1, 2)
average_ds = np.cumsum(d_s)[1::2] / points
average_as = np.cumsum(a_s)[1::2] / points
average_cs = np.cumsum(c_s)[1::2] / points

plt.axhline(y=73*np.pi**10. / 6842880., color="k", linestyle="dashed")
plt.plot(points, average_ds, color="r", label="Direct")
plt.plot(points, average_as, color="g", label="Antithetic")
plt.plot(points, average_cs, color="b", label="Control")
plt.xscale("log", base=2)
plt.xlim(2, n)
plt.xlabel(r"$n$")
plt.ylabel(r"$\bar{f}$")
plt.title(r"$\int_0^1\ldots\int_0^1 \frac{1}{1+x_1\ldots x_{10}}\,dx_1\ldots dx_{10}$")
plt.legend()
plt.savefig("(1+X1...X10)^(-1).pdf", bbox_inches='tight')
plt.clf()
