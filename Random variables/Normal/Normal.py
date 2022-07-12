"""An Investigation of Markov Chain Monte Carlo Methods: Normal.

Samples from normal distributions of various standard deviations using
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


def Met_Normal(σ, n, δ, start):
    """Metropolis normal distribution.

    Sample from a normal distribution (μ = 0) with given standard deviation,
    number of sample points, step size parameter, and starting point.
    """
    m_values = np.empty(n)
    m_values[0] = start
    for i in range(1, n):
        x = m_values[i-1]
        y = x + np.random.uniform(-δ, δ)
        if np.random.uniform(0, 1) < np.exp((x**2. - y**2.) / (2. * σ**2.)):
            m_values[i] = y
        else:
            m_values[i] = x
    return m_values


def f1(x):
    """Calculate the value of the random variable needed for f_1."""
    return .5 * np.exp(-abs(x))


K = 20
n = 2 ** K
ss = np.array([.1,
               10. ** (-.5),
               1.,
               10. ** .5,
               10.])
s_strings = [r"10^{-1}",
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


ms = np.ndarray((len(ss), len(ds), n))
for i in range(len(ss)):
    for j in range(len(ds)):
        ms[i, j] = Met_Normal(ss[i], n, ds[j], 0.)
        plt.hist(ms[i, j],
                 bins=np.arange(min(ms[i, j]), [max(ms[i, j])] + .01*ss[i],
                                .01 * ss[i]),
                 rwidth=1., align="left", color=colours[j])
        plt.title(r"Normal ($\sigma=%s$, $\delta=%s$)" %
                  (s_strings[i], d_strings[j]))
        plt.savefig("Normal distribution (σ = %s, δ = %s).pdf" % (ss[i],
                                                                  ds[j]))
        plt.clf()


K = 24
n = 2 ** K
factors = np.empty(K)
for i in range(K):
    factors[i] = 2 ** i
(means, means_nerror, f1s, f1s_nerror) = (np.empty((len(ss), len(ds))) for a in
                                          range(4))
(mean_mses, mean_mses_error, f1_mses, f1_mses_error) = (
    np.empty((len(ss), len(ds), K)) for a in range(4))
for i in range(len(ss)):
    for j in range(len(ds)):
        temp_ms = Met_Normal(ss[i], n, ds[j], 0.)
        means[i, j], means_nerror[i, j] = mean_error(temp_ms)
        mean_mses[i, j, 0] = means_nerror[i, j] ** 2.
        mean_mses_error[i, j, 0] = mean_mses[i, j, 0] * (2. / (n - 1.)) ** .5

        if ss[i] == 1.:
            temp_f1s = f1(temp_ms)
            f1s[i, j], f1s_nerror[i, j] = mean_error(temp_f1s)
            f1_mses[i, j, 0] = f1s_nerror[i, j] ** 2.
            f1_mses_error[i, j, 0] = f1_mses[i, j, 0] * (2. / (n - 1.)) ** .5

        for k in range(1, K):
            bins = int(len(temp_ms) / 2)
            binned = np.empty(bins)
            for m in range(bins):
                binned[m] = .5 * (temp_ms[2 * m] + temp_ms[2*m + 1])
            temp_ms = binned
            mean_mses[i, j, k] = variance(temp_ms) / bins
            mean_mses_error[i, j, k] = mean_mses[i, j, k] * (2./(bins-1.))**.5

            if ss[i] == 1.:
                temp_f1s = f1(temp_ms)
                f1_mses[i, j, k] = variance(temp_f1s) / bins
                f1_mses_error[i, j, k] = f1_mses[i, j, k] * (2./(bins-1.))**.5

        plt.errorbar(factors, mean_mses[i, j], yerr=mean_mses_error[i, j],
                     color=colours[j])
        plt.xscale("log", base=2)
        plt.xlim(1., factors[-1])
        plt.xlabel(r"$b$")
        plt.ylabel(r"Var($\bar{X}$)")
        plt.title(r"Normal binned MSE($\bar{\mu}$) ($\sigma=%s$, $\delta=%s$)"
                  % (s_strings[i], d_strings[j]))
        plt.savefig("Normal binned MSE(μ) (σ = %s, δ = %s).pdf" % (ss[i],
                                                                   ds[j]))
        plt.clf()

        if ss[i] == 1.:
            plt.errorbar(factors, f1_mses[i, j], yerr=f1_mses_error[i, j],
                         color=colours[j])
            plt.xscale("log", base=2)
            plt.xlim(1., factors[-1])
            plt.xlabel(r"$b$")
            plt.ylabel(r"Var($\bar{f}_1$)")
            plt.title(
                r"Normal binned MSE($\bar{f}_1$) ($\sigma=%s$, $\delta=%s$)" %
                (s_strings[i], d_strings[j]))
            plt.savefig("Normal binned MSE(f1) (σ = %s, δ = %s).pdf" % (ss[i],
                                                                        ds[j]))
            plt.clf()


ss = np.array([1.])
s_strings = [r"1"]
runs = 5
starts = np.linspace(0., 5., 21)
negs, negs_err, ones, ones_err, twos, twos_err, thrs, thrs_err = (
    np.ndarray((len(ss), len(ds), len(starts))) for a in range(8))
for i in range(len(ss)):
    s_starts = starts * ss[i]
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
                    if x <= 0.:
                        neg_times.append(m)
                    if abs(x) <= ss[i]:
                        one_times.append(m)
                        two_times.append(m)
                        thr_times.append(m)
                    elif abs(x) <= 2. * ss[i]:
                        two_times.append(m)
                        thr_times.append(m)
                    elif abs(x) <= 3. * ss[i]:
                        thr_times.append(m)
                    y = x + (2. * ds[j] * (np.random.uniform(0, 1) - .5))
                    if np.random.uniform(0, 1) < np.exp((x**2. - y**2.)
                                                        / (2. * ss[i]**2.)):
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
    plt.xlim(0., 5. * ss[i])
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Normal $t_{<}$ ($\sigma=%s$)" % s_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Normal t_less (σ = %s).pdf" % ss[i], bbox_inches="tight")
    plt.clf()
    for j in range(len(ds)):
        plt.errorbar(s_starts, ones[i, j], yerr=ones_err[i, j],
                     color=colours[j], label=r"$\delta=%s$" % d_strings[j])
    plt.xlim(ss[i], 5. * ss[i])
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Normal $t_1$ ($\sigma=%s$)" % s_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Normal t1 (σ = %s).pdf" % ss[i], bbox_inches="tight")
    plt.clf()
    for j in range(len(ds)):
        plt.errorbar(s_starts, twos[i, j], yerr=twos_err[i, j],
                     color=colours[j], label=r"$\delta=%s$" % d_strings[j])
    plt.xlim(2. * ss[i], 5. * ss[i])
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Normal $t_2$ ($\sigma=%s$)" % s_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Normal t2 (σ = %s).pdf" % ss[i], bbox_inches="tight")
    plt.clf()
    for j in range(len(ds)):
        plt.errorbar(s_starts, thrs[i, j], yerr=thrs_err[i, j],
                     color=colours[j], label=r"$\delta=%s$" % d_strings[j])
    plt.xlim(3. * ss[i], 5. * ss[i])
    plt.xlabel(r"$x_1$")
    plt.yscale("log")
    plt.ylabel(r"$t$")
    plt.title(r"Normal $t_3$ ($\sigma=%s$)" % s_strings[i])
    plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.savefig("Normal t3 (σ = %s).pdf" % ss[i], bbox_inches="tight")
    plt.clf()


mean_indexes = np.array([[16, 15, 15, 13, 11, 14, 11, 16, 16, 16],
                        [18, 14, 14, 10, 13, 14, 12, 10, 15, 14],
                        [22, 19, 15, 14, 14, 13, 14, 13, 15, 12],
                        [0, 0, 16, 18, 13, 11, 9, 9, 14, 14],
                        [0, 0, 21, 17, 17, 15, 13, 13, 11, 15]])
f1_indexes = np.array([22, 20, 17, 16, 21, 17, 18, 16, 18, 18])
mean_tau = np.zeros((5, 10))
for i in range(5):
    for j in range(10):
        a = mean_indexes[i, j]
        mean_tau[i, j] = mean_mses[i, j, a] / mean_mses[i, j, 0]
f1_tau = np.zeros(10)
for j in range(10):
    a = f1_indexes[j]
    f1_tau[j] = f1_mses[2, j, a] / f1_mses[2, j, 0]

plt.plot(ds, mean_tau[0], "-o", color=o_colours[0], label=r"$\sigma=%s$" %
         s_strings[0])
plt.plot(ds, mean_tau[1], "-o", color=o_colours[1], label=r"$\sigma=%s$" %
         s_strings[1])
plt.plot(ds, mean_tau[2], "-o", color=o_colours[2], label=r"$\sigma=%s$" %
         s_strings[2])
plt.plot(ds[2:], mean_tau[3][2:], "-o", color=o_colours[3],
         label=r"$\sigma=%s$" % s_strings[3])
plt.plot(ds[2:], mean_tau[4][2:], "-o", color=o_colours[4],
         label=r"$\sigma=%s$" % s_strings[4])
plt.xscale("log")
plt.yscale("log")
plt.xlim(ds[0], ds[-1])
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\tau$")
plt.ylabel(r"$\tau$")
plt.title(r"Normal $\mu$ correlation times")
plt.legend(fontsize="small")
plt.savefig("Normal μ correlation times.pdf")
plt.clf()


means_terror = means_nerror * np.sqrt(mean_tau)
plt.plot(ds, means_terror[0], "-o", color=o_colours[0], label=r"$\sigma=%s$"
         % s_strings[0])
plt.plot(ds, means_terror[1], "-o", color=o_colours[1], label=r"$\sigma=%s$"
         % s_strings[1])
plt.plot(ds, means_terror[2], "-o", color=o_colours[2], label=r"$\sigma=%s$"
         % s_strings[2])
plt.plot(ds[2:], means_terror[3][2:], "-o", color=o_colours[3],
         label=r"$\sigma=%s$" % s_strings[3])
plt.plot(ds[2:], means_terror[4][2:], "-o", color=o_colours[4],
         label=r"$\sigma=%s$" % s_strings[4])
plt.xscale("log")
plt.yscale("log")
plt.xlim(ds[0], ds[-1])
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\Delta\mu$")
plt.title(r"Normal $\Delta\mu$")
plt.legend(fontsize="small")
plt.savefig("Normal Δμ.pdf")
plt.clf()


f1s_terror = f1s_nerror * np.sqrt(f1_tau)
plt.plot(ds, f1_tau, "-o", color="navy", label=r"$\tau_{int,\bar{f}_1}$")
plt.plot(ds, f1s_terror[2], "-o", color="maroon", label=r"$\Delta\bar{f}_1$")
plt.xscale("log")
plt.yscale("log")
plt.xlim(ds[0], ds[-1])
plt.xlabel(r"$\delta$")
plt.title(r"Normal $f_1$ correlation times and $\Delta f_1$")
plt.legend()
plt.savefig("Normal f1 correlation times and Δf1.pdf")
plt.clf()


optimal_mean_ds = np.array([ds[3], ds[4], ds[5], ds[6], ds[7]])
optimal_mean_ds_error = np.array([[ds[3]-ds[2], ds[4]-ds[3], ds[5]-ds[4],
                                   ds[6]-ds[5], ds[7]-ds[6]],
                                  [ds[4]-ds[3], ds[5]-ds[4], ds[6]-ds[5],
                                   ds[7]-ds[6], ds[8]-ds[7]]])
plt.plot([.1, 10.], [optimal_mean_ds[0], optimal_mean_ds[-1]], linestyle="--",
         color="k", label=r"$\sqrt{10}\,\sigma$")
plt.errorbar(ss, optimal_mean_ds, yerr=optimal_mean_ds_error, fmt="o",
             color="m", label=r"$\delta_{opt}$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$\delta$")
plt.title(r"Optimal $\delta$ vs $\sigma$")
plt.legend()
plt.savefig("Normal optimal.pdf")
plt.clf()
