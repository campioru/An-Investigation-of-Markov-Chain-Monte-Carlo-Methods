"""An Investigation of Markov Chain Monte Carlo Methods: e^X.

Employs the antithetic variable and control variate approaches to reduce the
variance of the estimate of the integral of e^X from 0 to 1, with X = X1 and
X = X1...X10.

@author: Ruaidhrí Campion
"""

import numpy as np
import matplotlib.pyplot as plt


def mean(array):
    """Calculate the mean of an array."""
    total = 0.
    for i in array:
        total += i
    return total / len(array)


def average_list(array):
    """Calculate the evolution of the mean of an array."""
    result = np.zeros(len(array))
    result[0] = array[0]
    for i in range(1, len(array)):
        result[i] = ((result[i-1] * i) + array[i]) / (i+1.)
    return result


def variance(array):
    """Calculate the variance of an array."""
    mean_ = mean(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean_) ** 2.
    return total / (len(array) - 1.)


K = 20
n = 2 ** K
q = np.e - 1.
factors = np.empty(K)
for k in range(K):
    factors[k] = 2 ** (k+1)

(average_ds, average_as, average_cs) = (np.empty(K) for _ in range(3))
(xs, xs_) = (np.empty(n) for _ in range(2))
for i in range(int(n/2)):
    u = np.random.uniform(0, 1)
    xs[2*i] = u
    xs[2*i + 1] = np.random.uniform(0, 1)
    xs_[2*i] = u
    xs_[2*i + 1] = 1. - u
d_s = np.exp(xs)
a_s = np.exp(xs_)

mu = 0.5
var = 1. / 12.
cov = mean(xs * d_s) - mu * mean(d_s)
cstar = -cov / var
c_s = d_s + cstar * (xs - mu)

average_ds1 = average_list(d_s)
average_as1 = average_list(a_s)
average_cs1 = average_list(c_s)
for k in range(K):
    average_ds[k] = average_ds1[2**(k+1)-1]
    average_as[k] = average_as1[2**(k+1)-1]
    average_cs[k] = average_cs1[2**(k+1)-1]

plt.axhline(y=q, color="k", linestyle="dashed")
plt.plot(factors, average_ds, color="r", label="Direct")
plt.plot(factors, average_as, color="g", label="Antithetic")
plt.plot(factors, average_cs, color="b", label="Control")
plt.xscale("log", base=2)
plt.xlim(2., n)
plt.xlabel(r"$n$")
plt.ylabel(r"$\bar{f}$")
plt.title(r"$\int_0^1 e^x\,dx$")
plt.legend()
plt.savefig("e^(X1).pdf", bbox_inches='tight')
plt.clf()


K = 20
n = 2 ** K
factors = np.empty(K)
for k in range(K):
    factors[k] = 2 ** (k+1)
q = 10

(average_ds, average_as, average_cs) = (np.empty(K) for _ in range(3))
(xs, xs_) = (np.random.uniform(0, 1, (q, n)) for _ in range(2))
for i in range(int(n/2)):
    u = np.random.uniform(0, 1, q)
    for j in range(q):
        xs[j, 2*i] = u[j]
        xs[j, 2*i + 1] = np.random.uniform(0, 1)
        xs_[j, 2*i] = u[j]
        xs_[j, 2*i + 1] = 1. - u[j]
(Xs, Xs_) = (np.empty(n) for _ in range(2))
for j in range(n):
    v = 1.
    w = 1.
    for m in range(q):
        v *= xs[m, j]
        w *= xs_[m, j]
    Xs[j] = v
    Xs_[j] = w
d_s = np.exp(Xs)
a_s = np.exp(Xs_)

mu = 2. ** (-q)
var = variance(Xs)
cov = mean(Xs * d_s) - mu * mean(d_s)
cstar = -cov / var
c_s = d_s + cstar * (Xs - mu)

average_ds1 = average_list(d_s)
average_as1 = average_list(a_s)
average_cs1 = average_list(c_s)
for k in range(K):
    average_ds[k] = average_ds1[2**(k+1)-1]
    average_as[k] = average_as1[2**(k+1)-1]
    average_cs[k] = average_cs1[2**(k+1)-1]

plt.plot(factors, average_ds, color="r", label="Direct")
plt.plot(factors, average_as, color="g", label="Antithetic")
plt.plot(factors, average_cs, color="b", label="Control")
plt.xscale("log", base=2)
plt.xlim(2., n)
plt.xlabel(r"$n$")
plt.ylabel(r"$\bar{f}$")
plt.title(r"$\int_0^1\ldots\int_0^1 e^{x_1\ldots x_{10}}\,dx_1\ldots dx_{10}$")
plt.legend()
plt.savefig("e^(X1...X10).pdf", bbox_inches='tight')
plt.clf()
