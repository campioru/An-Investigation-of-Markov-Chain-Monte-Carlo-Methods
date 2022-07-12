"""An Investigation of Markov Chain Monte Carlo Methods: Exponential.

Samples from exponential distributions of various rate parameters using
Metropolis-Hastings algorithms of various step size parameters. Determines
the relationship between these parameters in terms of thermalisation and
integrated correlation time estimates.

@author: Ruaidhrí Campion
"""

import numpy as np
import matplotlib.pyplot as plt


def average(array):
    """Calculate the mean of an array."""
    total = 0.
    for i in array:
        total += i
    return total / len(array)


def variance(array):
    """Calculate the sample variance of an array."""
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    return total / (len(array) - 1.)


def mean_error(array):
    """Calculate the mean and corresponding error of an array."""
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    return mean, (total / ((len(array) - 1.) * (len(array)))) ** .5


def variance_error(array):
    """Calculate the sample variance and corresponding error of an array."""
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    total /= (len(array) - 1.)
    return total, total * (2. / (len(array) - 1.)) ** .5


def m_e_v_e(array):
    """Mean, sample variance, and errors.

    Calculate the mean, sample variance, and corresponding errors of an
    array.
    """
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    variance = total / (len(array) - 1.)
    return mean, (variance / len(array)) ** .5, variance, variance * (
        2. / (len(array) - 1.)) ** .5


def Met_Exp(λ, n, δ, start):
    """Metropolis exponential distribution.

    Sample from an exponential distribution (μ = 0) with given rate parameter,
    number of sample points, step size parameter, and starting point.
    """
    m_values = np.zeros(n)
    m_values[0] = start
    for i in range(1, n):
        x = m_values[i-1]
        y = abs(x + np.random.uniform(-δ, δ))
        if np.random.uniform(0, 1) < np.exp(λ * (x-y)):
            m_values[i] = y
        else:
            m_values[i] = x
    return m_values


def f2(x):
    """Calculate the value of the random variable needed for f_2."""
    return 2. * np.exp(-.5 * x ** 2.) / ((2. * np.pi) ** .5)


K = 20
n = 2 ** K
ls = np.array([.1,
               10. ** (-.5),
               1.,
               10. ** .5,
               10.])
l_strings = [r"10^{-1}",
             r"10^{-.5}",
             r"10^0",
             r"10^{.5}",
             r"10^1"]
ds = np.array([.01,
               10. ** (-1.5),
               .1,
               10. ** (-.5),
               1.,
               10. ** .5,
               10.,
               10. ** 1.5,
               100.,
               10. ** 2.5])
d_strings = [r"10^{-2}",
             r"10^{-1.5}",
             r"10^{-1}",
             r"10^{-.5}",
             r"10^0",
             r"10^{.5}",
             r"10^1",
             r"10^{1.5}",
             r"10^2",
             r"10^{2.5}"]
colours = ["r",
           "darkorange",
           "y",
           "g",
           "b",
           "purple",
           "k",
           "saddlebrown",
           "c",
           "violet",
           "dimgray"]
o_colours = ["#3e76ec",
             "#000000",
             "#ff0000",
             "#ffce01",
             "#179a13"]


ms = np.ndarray((len(ls), len(ds), n))
for i in range(len(ls)):
    for j in range(len(ds)):
        ms[i, j] = Met_Exp(ls[i], n, ds[j], 1. / ls[i])
        plt.hist(ms[i, j],
                 bins=np.arange(min(ms[i, j]), [max(ms[i, j])] + .01/ls[i],
                                .01 / ls[i]),
                 rwidth=1., align="left", color=colours[j])
        plt.xlim(left=0.)
        plt.title(r"Exponential ($\lambda=%s$, $\delta=%s$)" %
                  (l_strings[i], d_strings[j]))
        plt.savefig("Exponential distribution (λ = %s, δ = %s).pdf" % (ls[i],
                                                                       ds[j]))
        plt.clf()


K = 24
n = 2 ** K
factors = np.empty(K)
for i in range(K):
    factors[i] = 2 ** i
(means, means_nerror, f2s, f2s_nerror) = (np.empty((len(ls), len(ds))) for a in
                                          range(4))
(mean_mses, mean_mses_error, f2_mses, f2_mses_error) = (
    np.empty((len(ls), len(ds), K)) for a in range(4))
for i in range(len(ls)):
    for j in range(len(ds)):
        temp_ms = Met_Exp(ls[i], n, ds[j], 1. / ls[i])
        means[i, j], means_nerror[i, j] = mean_error(temp_ms)
        mean_mses[i, j, 0] = means_nerror[i, j] ** 2.
        mean_mses_error[i, j, 0] = mean_mses[i, j, 0] * (2. / (n - 1.)) ** .5

        if ls[i] == 1.:
            temp_f2s = f2(temp_ms)
            f2s[i, j], f2s_nerror[i, j] = mean_error(temp_f2s)
            f2_mses[i, j, 0] = f2s_nerror[i, j] ** 2.
            f2_mses_error[i, j, 0] = f2_mses[i, j, 0] * (2. / (n - 1.)) ** .5

        for k in range(1, K):
            bins = int(len(temp_ms) / 2)
            binned = np.empty(bins)
            for m in range(bins):
                binned[m] = .5 * (temp_ms[2 * m] + temp_ms[2*m + 1])
            temp_ms = binned
            mean_mses[i, j, k] = variance(temp_ms) / bins
            mean_mses_error[i, j, k] = mean_mses[i, j, k] * (2./(bins-1.))**.5

            if ls[i] == 1.:
                temp_f2s = f2(temp_ms)
                f2_mses[i, j, k] = variance(temp_f2s) / bins
                f2_mses_error[i, j, k] = f2_mses[i, j, k] * (2./(bins-1.))**.5

        plt.errorbar(factors, mean_mses[i, j], yerr=mean_mses_error[i, j],
                     color=colours[j])
        plt.xscale("log", base=2)
        plt.xlim(1., factors[-1])
        plt.xlabel(r"$b$")
        plt.ylabel(r"Var($\bar{X}$)")
        plt.title(r"Exponential binned MSE($\bar{\mu}$) \
                  ($\lambda=%s$, $\delta=%s$)" %
                  (l_strings[i], d_strings[j]))
        plt.savefig("Exponential binned MSE(μ) (λ = %s, δ = %s).pdf" % (ls[i],
                                                                        ds[j]))
        plt.clf()

        if ls[i] == 1.:
            plt.errorbar(factors, f2_mses[i, j], yerr=f2_mses_error[i, j],
                         color=colours[j])
            plt.xscale("log", base=2)
            plt.xlim(1., factors[-1])
            plt.xlabel(r"$b$")
            plt.ylabel(r"Var($\bar{f}_2$)")
            plt.title(r"Exponential binned MSE($\bar{f}_2$) \
                      ($\lambda=%s$, $\delta=%s$)" %
                      (l_strings[i], d_strings[j]))
            plt.savefig("Exponential binned MSE(f2) (λ = %s, δ = %s).pdf" %
                        (ls[i], ds[j]))
            plt.clf()


ls = np.array([1.])
l_strings = [r"1"]
runs = 5
starts = np.linspace(1., 6., 21)
negs, negs_err, ones, ones_err, twos, twos_err, thrs, thrs_err = (
    np.ndarray((len(ls), len(ds), len(starts))) for a in range(8))
for i in range(len(ls)):
    sigma = 1. / ls[i]
    s_starts = starts * sigma
    for j in range(len(ds)):
        for k in range(len(starts)):
            neg_time, one_time, two_time, thr_time = (np.empty(runs) for a in
                                                      range(4))
            for L in range(runs):
                x = s_starts[k]
                neg_times, one_times, two_times, thr_times = ([] for a in
                                                              range(4))
                m = 0
                while (len(neg_times) == 0 or len(one_times) == 0
                        or len(two_times) == 0 or len(thr_times) == 0):
                    m += 1
                    if x <= sigma:
                        neg_times.append(m)
                    if abs(x - sigma) <= sigma:
                        one_times.append(m)
                        two_times.append(m)
                        thr_times.append(m)
                    elif abs(x - sigma) <= 2. * sigma:
                        two_times.append(m)
                        thr_times.append(m)
                    elif abs(x - sigma) <= 3. * sigma:
                        thr_times.append(m)
                    y = abs(x + np.random.uniform(-ds[j], ds[j]))
                    if np.random.uniform(0, 1) < np.exp(ls[i] * (x - y)):
                        x = y
                neg_time[L] = neg_times[0]
                one_time[L] = one_times[0]
                two_time[L] = two_times[0]
                thr_time[L] = thr_times[0]
            negs[i, j, k], negs_err[i, j, k] = mean_error(neg_time)
            ones[i, j, k], ones_err[i, j, k] = mean_error(one_time)
            twos[i, j, k], twos_err[i, j, k] = mean_error(two_time)
            thrs[i, j, k], thrs_err[i, j, k] = mean_error(thr_time)

    for j in range(len(ds)):
        plt.errorbar(s_starts, negs[i, j], yerr=negs_err[i, j],
                     color=colours[j], label=r"$\delta=%s$" % d_strings[j])
    plt.xlim(sigma, 6. * sigma)
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Exponential $t_{<}$ ($\lambda=%s$)" % l_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Exponential t_less (λ = %s).pdf" % ls[i], bbox_inches="tight")
    plt.clf()
    for j in range(len(ds)):
        plt.errorbar(s_starts, ones[i, j], yerr=ones_err[i, j],
                     color=colours[j], label=r"$\delta=%s$" % d_strings[j])
    plt.xlim(2. * sigma, 6. * sigma)
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Exponential $t_1$ ($\lambda=%s$)" % l_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Exponential t1 (λ = %s).pdf" % ls[i], bbox_inches="tight")
    plt.clf()
    for j in range(len(ds)):
        plt.errorbar(s_starts, twos[i, j], yerr=twos_err[i, j],
                     color=colours[j], label=r"$\delta=%s$" % d_strings[j])
    plt.xlim(3. * sigma, 6. * sigma)
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Exponential $t_2$ ($\lambda=%s$)" % l_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Exponential t2 (λ = %s).pdf" % ls[i], bbox_inches="tight")
    plt.clf()
    for j in range(len(ds)):
        plt.errorbar(s_starts, thrs[i, j], yerr=thrs_err[i, j],
                     color=colours[j], label=r"$\delta=%s$" % d_strings[j])
    plt.xlim(4. * sigma, 6. * sigma)
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Exponential $t_3$ ($\lambda=%s$)" % l_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Exponential t3 (λ = %s).pdf" % ls[i], bbox_inches="tight")
    plt.clf()


mean_indexes = np.array([[0, 0, 22, 21, 15, 14, 10, 11, 12, 12],
                        [0, 18, 21, 15, 16, 12, 13, 12, 15, 11],
                        [20, 19, 17, 15, 11, 14, 13, 10, 11, 12],
                        [19, 19, 12, 14, 12, 9, 10, 14, 14, 14],
                        [16, 13, 11, 12, 12, 11, 12, 13, 14, 15]])
f2_indexes = np.array([20, 20, 17, 15, 13, 12, 12, 11, 12, 13])
mean_tau = np.zeros((5, 10))
for i in range(5):
    for j in range(10):
        a = mean_indexes[i, j]
        mean_tau[i, j] = mean_mses[i, j, a] / mean_mses[i, j, 0]
f2_tau = np.zeros(10)
for j in range(10):
    a = f2_indexes[j]
    f2_tau[j] = f2_mses[2, j, a] / f2_mses[2, j, 0]


plt.plot(ds[2:], mean_tau[0][2:], "-o", color=o_colours[0],
         label=r"$\lambda=%s$" % l_strings[0])
plt.plot(ds[1:], mean_tau[1][1:], "-o", color=o_colours[1],
         label=r"$\lambda=%s$" % l_strings[1])
plt.plot(ds, mean_tau[2], "-o", color=o_colours[2], label=r"$\lambda=%s$" %
         l_strings[2])
plt.plot(ds, mean_tau[3], "-o", color=o_colours[3], label=r"$\lambda=%s$" %
         l_strings[3])
plt.plot(ds, mean_tau[4], "-o", color=o_colours[4], label=r"$\lambda=%s$" %
         l_strings[4])
plt.xscale("log")
plt.yscale("log")
plt.xlim(ds[0], ds[-1])
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\tau$")
plt.title(r"Exponential $\mu$ correlation times")
plt.legend(fontsize="small")
plt.savefig("Exponential μ correlation times.pdf")
plt.clf()

means_terror = means_nerror * np.sqrt(mean_tau)
plt.plot(ds[2:], means_terror[0][2:], "-o", color=o_colours[0],
         label=r"$\lambda=%s$" % l_strings[0])
plt.plot(ds[1:], means_terror[1][1:], "-o", color=o_colours[1],
         label=r"$\lambda=%s$" % l_strings[1])
plt.plot(ds, means_terror[2], "-o", color=o_colours[2],
         label=r"$\lambda=%s$" % l_strings[2])
plt.plot(ds, means_terror[3], "-o", color=o_colours[3],
         label=r"$\lambda=%s$" % l_strings[3])
plt.plot(ds, means_terror[4], "-o", color=o_colours[4],
         label=r"$\lambda=%s$" % l_strings[3])
plt.xscale("log")
plt.yscale("log")
plt.xlim(ds[0], ds[-1])
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\Delta\mu$")
plt.title(r"Exponential $\Delta\mu$")
plt.legend(fontsize="small")
plt.savefig("Exponential Δμ.pdf")
plt.clf()

f2s_terror = f2s_nerror * np.sqrt(f2_tau)
plt.plot(ds, f2_tau, "-o", color="navy", label=r"$\tau_{int,\bar{f}_2}$")
plt.plot(ds, f2s_terror[2], "-o", color="maroon", label=r"$\Delta\bar{f}_2$")
plt.xscale("log")
plt.yscale("log")
plt.xlim(ds[0], ds[-1])
plt.xlabel(r"$\delta$")
plt.title(r"Exponential $f_2$ correlation times and $\Delta f_2$")
plt.legend()
plt.savefig("Exponential f2 correlation times and Δf2.pdf")
plt.clf()


optimal_mean_ds = np.array([ds[7], ds[6], ds[5], ds[4], ds[3]])
optimal_mean_ds_error = np.array([[ds[7]-ds[6], ds[6]-ds[5], ds[5]-ds[4],
                                   ds[4]-ds[3], ds[3]-ds[2]],
                                 [ds[8]-ds[7], ds[7]-ds[6], ds[6]-ds[5],
                                  ds[5]-ds[4], ds[4]-ds[3]]])
plt.plot([.1, 10.], [optimal_mean_ds[0], optimal_mean_ds[-1]], linestyle="--",
         color="k", label=r"$\frac{\sqrt{10}}{\lambda}$")
plt.errorbar(ls, optimal_mean_ds, yerr=optimal_mean_ds_error, fmt="o",
             color="m", label=r"$\delta_{opt}$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\delta$")
plt.title(r"Optimal $\delta$ vs $\lambda$")
plt.legend()
plt.savefig("Exponential optimal.pdf")
plt.clf()
