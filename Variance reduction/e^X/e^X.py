import numpy as np
import matplotlib.pyplot as plt

def average(array):
    total = 0.
    for i in array:
        total += i
    return total / len(array)

def average_list(array):
    result = np.zeros(len(array))
    result[0] = array[0]
    for i in range(1,len(array)):
        result[i] = ((result[i-1] * i) + array[i]) / (i+1.)
    return result

def variance(array):
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    return total / (len(array) - 1.)

def mean_error(array):
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    return mean, (total / ((len(array) - 1.) * (len(array)))) ** 0.5

def variance_error(array):
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    total /= (len(array) - 1.)
    return total, total * (2. / (len(array) - 1.)) ** 0.5

def m_e_v_e(array):
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    variance = total / (len(array) - 1.)
    return mean, (variance / len(array)) ** 0.5, variance, variance * (
        2. / (len(array) - 1.)) ** 0.5

def Simp(f, a, b, n):
    h = (b - a) / n
    even_total = 0.
    odd_total = 0.
    for j in range(1,int(n/2)+1):
        even_total += f(a + (2. * j - 1.) * h)
    for j in range(1, int(n/2)):
        odd_total += f(a + 2. * j * h)
    return (f(a) + f(b) + 4. * even_total + 2. * odd_total) * (h / 3.)

def f(x):
    return np.exp(x)

K = 20
n = 2 ** K
q = np.e - 1.
factors = np.empty(K)
for k in range(K):
    factors[k] = 2 ** (k+1)

(average_ds, average_as, average_cs, average_ss) = (np.empty(K) for a in range(4))
xs = np.empty(n)
xs_ = np.empty(n)
for i in range(int(n/2)):
    u = np.random.uniform(0,1)
    xs[2*i] = u
    xs[2*i + 1] = np.random.uniform(0,1)
    xs_[2*i] = u
    xs_[2*i + 1] = 1. - u
d_s = np.exp(xs)
a_s = np.exp(xs_)

mu = 0.5
var = 1. / 12.
cov = average(xs * d_s) - mu * average(d_s)
cstar = -cov / var
c_s = d_s + cstar * (xs - mu)

average_ds1 = average_list(d_s)
average_as1 = average_list(a_s)
average_cs1 = average_list(c_s)
# average_ss[k] = Simp(f,0,1,n)
for k in range(K):
    average_ds[k] = average_ds1[2**(k+1)-1]
    average_as[k] = average_as1[2**(k+1)-1]
    average_cs[k] = average_cs1[2**(k+1)-1]

plt.axhline(y = q, color = "k", linestyle = "dashed")
plt.plot(factors, average_ds, color = "r", label = "Direct")
plt.plot(factors, average_as, color = "g", label = "Antithetic")
plt.plot(factors, average_cs, color = "b", label = "Control")
# plt.plot(factors, average_ss, color = "m", label = "Simpson")
plt.xscale("log", base = 2)
plt.xlim(2.,n)
plt.xlabel(r"$n$")
plt.ylabel(r"$\bar{f}$")
plt.title(r"$\int_0^1 e^x\,dx$")
plt.legend()
plt.savefig("e1.pdf", bbox_inches='tight')
plt.show()



K = 20
n = 2 ** K
factors = np.empty(K)
for k in range(K):
    factors[k] = 2 ** (k+1)
q = 10

(average_ds, average_as, average_cs) = (np.empty(K) for a in range(3))
xs = np.random.uniform(0,1,(q,n))
xs_ = np.random.uniform(0,1,(q,n))
for i in range(int(n/2)):
    u = np.random.uniform(0,1,q)
    for j in range(q):
        xs[j,2*i] = u[j]
        xs[j,2*i + 1] = np.random.uniform(0,1)
        xs_[j,2*i] = u[j]
        xs_[j,2*i + 1] = 1. - u[j]
Xs = np.empty(n)
Xs_ = np.empty(n)
for j in range(n):
    v = 1.
    w = 1.
    for m in range(q):
        v *= xs[m,j]
        w *= xs_[m,j]
    Xs[j] = v
    Xs_[j] = w
d_s = np.exp(Xs)
a_s = np.exp(Xs_)

mu = 2. ** (-q)
var = variance(Xs)
cov = average(Xs * d_s) - mu * average(d_s)
cstar = -cov / var
c_s = d_s + cstar * (Xs - mu)

average_ds1 = average_list(d_s)
average_as1 = average_list(a_s)
average_cs1 = average_list(c_s)
# average_ss[k] = Simp(f,0,1,n)
for k in range(K):
    average_ds[k] = average_ds1[2**(k+1)-1]
    average_as[k] = average_as1[2**(k+1)-1]
    average_cs[k] = average_cs1[2**(k+1)-1]

plt.plot(factors, average_ds, color = "r", label = "Direct")
plt.plot(factors, average_as, color = "g", label = "Antithetic")
plt.plot(factors, average_cs, color = "b", label = "Control")
plt.xscale("log", base = 2)
plt.xlim(2.,n)
plt.xlabel(r"$n$")
plt.ylabel(r"$\bar{f}$")
plt.title(r"$\int_0^1\ldots\int_0^1 e^{x_1\ldots x_{10}}\,dx_1\ldots dx_{10}$")
plt.legend()
plt.savefig("e2.pdf", bbox_inches='tight')
plt.show()