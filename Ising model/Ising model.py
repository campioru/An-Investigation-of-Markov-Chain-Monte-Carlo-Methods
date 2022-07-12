"""An Investigation of Markov Chain Monte Carlo Methods: Ising model.

Simulates Ising models with nearest neighbour interactions and no external
magentic field, using the Metropolis-Hastings and heat bath algorithms.
Determines energy, magnetisation, and heat capacity, and their corresponding
integrated correlation times using the binning method.

In this project, k_B = g = 1 for convenience, corresponding to a temperature T
representing k_B * T, an energy U representing U / g, and a heat capacity c_V
representing c_V / k_B.

@author: Ruaidhrí Campion
"""

import numpy as np
import matplotlib.pyplot as plt


def corr(spins):
    """Calculate the spin-spin correlation of a given state of the system."""
    eye = np.shape(spins)[0]
    jay = np.shape(spins)[1]
    total = 0.
    for A in range(eye):
        for B in range(jay):
            total += spins[A, B] * (spins[(A+1) % eye, B]
                                    + spins[A, (B+1) % jay])
    return total


def mag(spins):
    """Calculate the total magnetisation of a given state of the system."""
    total = 0.
    for A in spins:
        for B in A:
            total += B
    return total


def mean(array):
    """Calculate the mean of an array."""
    total = 0.
    for i in array:
        total += i
    return total / len(array)


def variance(array):
    """Calculate the variance of an array."""
    mean_ = mean(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean_) ** 2.
    return total / (len(array) - 1.)


def mean_error(array):
    """Calculate the mean and the corresponding error of an array."""
    mean_ = mean(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean_) ** 2.
    return mean_, (total / ((len(array) - 1.) * (len(array)))) ** .5


def m_e_v_e(array):
    """Calculate the mean and variance and corresponding errors of an array."""
    mean_ = mean(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean_) ** 2.
    variance = total / (len(array) - 1.)
    return (mean_, (variance / len(array)) ** .5, variance,
            variance * (2. / (len(array) - 1.)) ** .5)


def Ising(alg, width, length, g, T, n, equib):
    """Ising model.

    Run a Metropolis-Hastings (alg = 1) or heat bath (alg = 2) simulation of an
    Ising model of a given size with a given interaction term at a given
    temperature for a given number of samples.

    Includes an extra parameter that determines if the algorithm starts in a
    random state (equib = False) or an equilibrium state for temperatures less
    than the critical temperature (equib = True).

    Returns the final energy, final magnetisation, and the initial and final
    states.
    """
    spins = np.ones((n, width, length))
    if equib is False:
        for i in range(width):
            for j in range(length):
                if np.random.uniform(0, 1) < .5:
                    spins[0, i, j] = -1.
    elif equib is True:
        if np.random.uniform(0, 1) < .5:
            spins *= -1.
        if g < 0.:
            for i in range(width):
                for j in range(length):
                    if ((i % 2 == 0 and j % 2 == 0)
                            or (i % 2 == 1 and j % 2 == 1)):
                        spins[0, i, j] *= -1.

    c = corr(spins[0])
    M = np.empty(n)
    m = mag(spins[0])
    M[0] = m
    U = np.empty(n)
    E = -g * c
    U[0] = E

    if alg == 1:
        for t in range(1, n):
            spins[t] = spins[t-1]
            for i in range(width):
                for j in range(length):
                    c_diff = -2.*spins[t, i, j] * (
                        spins[t, (i+1) % width, j]
                        + spins[t, i, (j+1) % length]
                        + spins[t, (i-1) % width, j]
                        + spins[t, i, (j-1) % length])
                    m_diff = -2. * spins[t, i, j]
                    E_diff = -g * c_diff
                    if (E_diff <= 0.
                            or np.random.uniform(0, 1) <= np.exp(-E_diff / T)):
                        spins[t, i, j] *= -1.
                        c += c_diff
                        m += m_diff
                        E += E_diff
            M[t] = m
            U[t] = E
    elif alg == 2:
        for t in range(1, n):
            spins[t] = spins[t-1]
            for i in range(width):
                for j in range(length):
                    c_diff = -2.*spins[t, i, j] * (
                        spins[t, (i+1) % width, j]
                        + spins[t, i, (j+1) % length]
                        + spins[t, (i-1) % width, j]
                        + spins[t, i, (j-1) % length])
                    m_diff = -2. * spins[t, i, j]
                    E_diff = -g * c_diff
                    p = np.exp(-E_diff/T)
                    if np.random.uniform(0, 1) <= p / (p+1.):
                        spins[t, i, j] *= -1.
                        c += c_diff
                        m += m_diff
                        E += E_diff
            M[t] = m
            U[t] = E

    return U / (width*length), M / (width*length), spins[0], spins[-1]


algs = [1, 2]
width, length = (32 for _ in range(2))
gs = [1., -1.]
T_crit = 2. / (np.log(1. + 2.**.5))
Ts = np.concatenate((np.linspace(.2, .8, 4),
                     np.linspace(1., 1.9, 10),
                     np.linspace(2., 2.25, 6),
                     [2.26, T_crit, 2.28],
                     np.linspace(2.3, 2.45, 4),
                     np.linspace(2.5, 2.9, 5),
                     np.linspace(3., 3.8, 5),
                     np.linspace(4., 6., 9)))
K = 17
n = 2 ** K
disc = 2000

(Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror) = (
    np.empty((len(gs), len(algs), len(Ts))) for _ in range(6))
(U_mses, U_mses_error, M_mses, M_mses_error, V_mses) = (
    np.empty((len(gs), len(algs), len(Ts), K)) for _ in range(5))
init_spins, final_spins = (
    np.empty((len(gs), len(algs), len(Ts), width, length)) for _ in range(2))

for g in range(len(gs)):
    for a in range(len(algs)):
        for T in range(len(Ts)):
            temp_Us, temp_Ms, init_spins[g, a, T], final_spins[g, a, T] = (
                Ising(algs[a], width, length, gs[g], Ts[T], n, False))
            temp_Us = temp_Us[disc:]
            temp_Ms = temp_Ms[disc:]

            (Us[g, a, T], Us_nerror[g, a, T], Vs[g, a, T],
             Vs_nerror[g, a, T]) = m_e_v_e(temp_Us)
            U_mses[g, a, T, 0] = Us_nerror[g, a, T] ** 2.
            U_mses_error[g, a, T, 0] = U_mses[g, a, T, 0] * (2./(n - 1.)) ** .5
            V_mses[g, a, T, 0] = Vs_nerror[g, a, T] ** 2.

            Ms[g, a, T], Ms_nerror[g, a, T] = mean_error(temp_Ms)
            M_mses[g, a, T, 0] = Ms_nerror[g, a, T] ** 2.
            M_mses_error[g, a, T, 0] = M_mses[g, a, T, 0] * (2./(n - 1.)) ** .5

            for k in range(1, K):
                bins = int(len(temp_Us) / 2)
                binned_Us, binned_Ms = (np.empty(bins) for _ in range(2))
                for b in range(bins):
                    binned_Us[b] = .5 * (temp_Us[2*b] + temp_Us[2*b + 1])
                    binned_Ms[b] = .5 * (temp_Ms[2*b] + temp_Ms[2*b + 1])
                temp_Us = binned_Us
                temp_Ms = binned_Ms

                U_mses[g, a, T, k] = variance(temp_Us) / bins
                U_mses_error[g, a, T, k] = (U_mses[g, a, T, k]
                                            * (2. / (bins - 1.)) ** .5)
                V_mses[g, a, T, k] = U_mses_error[g, a, T, k] ** 2.

                M_mses[g, a, T, k] = variance(temp_Ms) / bins
                M_mses_error[g, a, T, k] = (M_mses[g, a, T, k]
                                            * (2. / (bins - 1.)) ** .5)


alg_strings = ["M-H", "HB"]
g_strings = ["1", "-1"]
T_strings = ["0.2", "0.4", "0.6", "0.8", "1", "1.1", "1.2", "1.3", "1.4",
             "1.5", "1.6", "1.7", "1.8", "1.9", "2", "2.05", "2.1", "2.15",
             "2.2", "2.25", "2.26", "T_c", "2.28", "2.3", "2.35", "2.4",
             "2.45", "2.5", "2.6", "2.7", "2.8", "2.9", "3", "3.2", "3.4",
             "3.6", "3.8", "4", "4.25", "4.5", "4.75", "5", "5.25", "5.5",
             "5.75", "6"]
T_ = len(Ts)
Tcolours = np.empty((T_, 3))
for i in range(T_):
    Tcolours[i, 0] = 1. - i/(T_-1.)
    Tcolours[i, 1] = 0.
    Tcolours[i, 2] = i/(T_-1.)
factors = np.empty(K)
for i in range(K):
    factors[i] = 2 ** i
b = "both"
f = False

for g in range(len(gs)):
    for a in range(len(algs)):
        for T in range(len(Ts)):
            plt.imshow(init_spins[g, a, T])
            plt.tick_params(axis=b, which=b, bottom=f, top=f, labelbottom=f,
                            left=f, right=f, labelleft=f)
            plt.title(r"Initial state (%s, $g=%s$, $T=%s$)" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.savefig("Initial (%s, g = %s, T = %s).pdf" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.clf()

            plt.imshow(final_spins[g, a, T])
            plt.tick_params(axis=b, which=b, bottom=f, top=f, labelbottom=f,
                            left=f, right=f, labelleft=f)
            plt.title(r"Final state (%s, $g=%s$, $T=%s$)" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.savefig("Final (%s, g = %s, T = %s).pdf" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.clf()

            plt.errorbar(factors, U_mses[g, a, T], yerr=U_mses_error[g, a, T],
                         color=(Tcolours[T, 0], Tcolours[T, 1],
                                Tcolours[T, 2]))
            plt.xscale("log", base=2)
            plt.xlim(1., factors[-1])
            plt.xlabel(r"$b$")
            plt.ylabel(r"Var($\bar{U}$)")
            plt.title(r"Binned MSE($\bar{U}$) (%s, $g=%s$, $T=%s$)" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.savefig("Binned U (%s, g = %s, T = %s).pdf" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.clf()

            plt.errorbar(factors, M_mses[g, a, T], yerr=M_mses_error[g, a, T],
                         color=(Tcolours[T, 0], Tcolours[T, 1],
                                Tcolours[T, 2]))
            plt.xscale("log", base=2)
            plt.xlim(1., factors[-1])
            plt.xlabel(r"$b$")
            plt.ylabel(r"Var($\bar{M}$)")
            plt.title(r"Binned MSE($\bar{M}$) (%s, $g=%s$, $T=%s$)" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.savefig("Binned M (%s, g = %s, T = %s).pdf" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.clf()

            plt.plot(factors, V_mses[g, a, T] / (Ts[T]*Ts[T]),
                     color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
            plt.xscale("log", base=2)
            plt.xlim(1., factors[-1])
            plt.xlabel(r"$b$")
            plt.ylabel(r"Var($\bar{c_V}$)")
            plt.title(r"Binned MSE($\bar{c_V}$) (%s, $g=%s$, $T=%s$)" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.savefig("Binned c_V (%s, g = %s, T = %s).pdf" % (
                alg_strings[a], g_strings[g], T_strings[T]))
            plt.clf()


Us_indexes, Ms_indexes = (np.empty((len(gs), len(algs), len(Ts)), dtype=int)
                          for _ in range(2))
Us_indexes[0, 0] = np.array([0, 0, 11, 8, 4, 4, 6, 4, 4, 4, 4, 6, 7, 7, 8, 9,
                             10, 9, 9, 10, 10, 9, 9, 9, 8, 9, 10, 7, 8, 6, 5,
                             7, 8, 8, 7, 6, 6, 7, 5, 6, 7, 8, 8, 8, 8, 6])
Ms_indexes[0, 0] = np.array([0, 0, 11, 8, 4, 4, 6, 4, 4, 4, 4, 5, 6, 7, 8, 9,
                             9, 9, 10, 13, 13, 13, 13, 12, 11, 10, 9, 10, 8, 8,
                             8, 7, 9, 6, 6, 8, 6, 7, 5, 3, 6, 7, 6, 6, 8, 7])
Us_indexes[0, 1] = np.array([0, 0, 10, 10, 3, 4, 3, 6, 6, 6, 8, 6, 6, 7, 8, 7,
                             8, 8, 10, 10, 12, 10, 10, 9, 11, 9, 9, 11, 9, 8,
                             7, 6, 8, 10, 7, 7, 5, 4, 6, 4, 6, 4, 5, 4, 5, 6])
Ms_indexes[0, 1] = np.array([0, 0, 10, 10, 3, 4, 3, 6, 5, 5, 8, 6, 6, 6, 9, 7,
                             9, 8, 10, 10, 12, 10, 10, 9, 12, 11, 10, 11, 9, 8,
                             10, 8, 8, 7, 6, 8, 6, 7, 5, 6, 5, 7, 7, 5, 5, 6])
Us_indexes[1, 0] = np.array([0, 0, 2, 4, 3, 4, 6, 3, 4, 5, 5, 4, 6, 7, 5, 5, 7,
                             7, 10, 9, 8, 9, 9, 9, 10, 8, 8, 8, 6, 5, 5, 5, 5,
                             6, 6, 6, 5, 5, 7, 6, 6, 6, 7, 5, 5, 6])
Ms_indexes[1, 0] = np.array([0, 0, 2, 4, 3, 4, 3, 4, 5, 5, 4, 5, 5, 5, 5, 5, 6,
                             5, 6, 6, 6, 5, 5, 5, 5, 6, 6, 7, 6, 7, 8, 6, 6, 6,
                             7, 6, 7, 6, 7, 6, 7, 7, 7, 7, 7, 6])
Us_indexes[1, 1] = np.array([0, 0, 5, 3, 2, 4, 3, 4, 4, 7, 5, 5, 6, 6, 6, 10,
                             7, 9, 9, 11, 9, 11, 9, 10, 10, 10, 10, 8, 7, 7, 7,
                             7, 6, 5, 5, 5, 5, 6, 4, 4, 4, 4, 5, 4, 4, 4])
Ms_indexes[1, 1] = np.array([0, 0, 4, 6, 3, 5, 4, 4, 5, 4, 4, 5, 5, 5, 5, 5, 6,
                             5, 6, 4, 5, 4, 6, 5, 5, 4, 6, 6, 6, 6, 5, 5, 6, 5,
                             7, 5, 5, 4, 4, 5, 4, 5, 6, 6, 6, 6])

Us_tau, Us_terror, Ms_tau, Ms_terror = (np.empty((len(gs), len(algs), len(Ts)))
                                        for _ in range(4))
for g in range(len(gs)):
    for a in range(len(algs)):
        for T in range(len(Ts)):
            i = U_mses[g, a, T, Us_indexes[g, a, T]]
            j = M_mses[g, a, T, Ms_indexes[g, a, T]]
            Us_terror[g, a, T] = i ** .5
            Ms_terror[g, a, T] = j ** .5
            Us_tau[g, a, T] = i / U_mses[g, a, T, 0]
            Ms_tau[g, a, T] = j / M_mses[g, a, T, 0]


rgbm = np.array([["r", "g"], ["b", "m"]])
lims = [2, 1]
for g in range(len(gs)):
    plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
    for a in range(len(algs)):
        plt.plot(Ts, Us_tau[g, a], color=rgbm[g, a], label=r"%s, $g=%s$" %
                 (alg_strings[a], g_strings[g]))
    plt.xlim(Ts[lims[g]], Ts[-1])
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\tau$")
    plt.title("Energy correlation times ($g=%s$)" % g_strings[g])
    plt.legend()
    plt.savefig("U times (g = %s).pdf" % g_strings[g])
    plt.clf()

    plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
    for a in range(len(algs)):
        plt.plot(Ts, Ms_tau[g, a], color=rgbm[g, a], label=r"%s, $g=%s$" %
                 (alg_strings[a], g_strings[g]))
    plt.xlim(Ts[lims[g]], Ts[-1])
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\tau$")
    plt.title("Magnetisation correlation times ($g=%s$)" % g_strings[g])
    plt.legend()
    plt.savefig("M times (g = %s).pdf" % g_strings[g])
    plt.clf()


for g in range(len(gs)):
    for a in range(len(algs)):
        plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
        plt.errorbar(Ts, Us[g, a], yerr=Us_terror[g, a], color=rgbm[g, a])
        plt.xlim(min(Ts), max(Ts))
        plt.xlabel(r"$T$")
        plt.ylabel(r"$U$")
        plt.title(r"Average energy (%s, $g=%s$)" % (alg_strings[a],
                                                    g_strings[g]))
        plt.legend()
        plt.savefig("U (%s, g = %s).pdf" % (alg_strings[a], g_strings[g]))
        plt.clf()

        plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
        plt.errorbar(Ts, abs(Ms[g, a]), yerr=Ms_terror[g, a], color=rgbm[g, a])
        plt.xlim(min(Ts), max(Ts))
        plt.xlabel(r"$T$")
        plt.ylabel(r"$|M|$")
        plt.title(r"Average magnetisation (%s, $g=%s$)" % (alg_strings[a],
                                                           g_strings[g]))
        plt.legend()
        plt.savefig("M (%s, g = %s).pdf" % (alg_strings[a], g_strings[g]))
        plt.clf()

        plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
        plt.plot(Ts, Vs[g, a] / (Ts*Ts), color=rgbm[g, a])
        plt.xlim(min(Ts), max(Ts))
        plt.xlabel(r"$T$")
        plt.ylabel(r"$c_V$")
        plt.title(r"Average heat capacity (%s, $g=%s$)" % (alg_strings[a],
                                                           g_strings[g]))
        plt.legend()
        plt.savefig("c_V (%s, g = %s).pdf" % (alg_strings[a], g_strings[g]))
        plt.clf()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
for g in range(len(gs)):
    for a in range(len(algs)):
        plt.errorbar(Ts, Us[g, a], yerr=Us_terror[g, a], color=rgbm[g, a],
                     label=r"%s, $g=%s$" % (alg_strings[a], g_strings[g]))
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
plt.title(r"Average energy")
plt.legend()
plt.savefig("U.pdf")
plt.clf()

for g in range(len(gs)):
    plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
    for a in range(len(algs)):
        plt.errorbar(Ts, abs(Ms[g, a]), yerr=Ms_terror[g, a], color=rgbm[g, a],
                     label=r"%s, $g=%s$" % (alg_strings[a], g_strings[g]))
    plt.xlabel(r"$T$")
    plt.ylabel(r"$|M|$")
    plt.title(r"Average magnetisation ($g=%s$)" % g_strings[g])
    plt.legend()
    plt.savefig("M (g = %s).pdf" % g_strings[g])
    plt.clf()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
for g in range(len(gs)):
    for a in range(len(algs)):
        plt.plot(Ts, Vs[g, a] / (Ts*Ts), color=rgbm[g, a], label=r"M-H, $g=1$")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$c_V$")
plt.title(r"Average heat capacity")
plt.legend()
plt.savefig("c_V.pdf")
plt.clf()
