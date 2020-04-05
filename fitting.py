# coding=utf-8
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import numpy as np
from matplotlib import pyplot
import datetime
import pandas as pd

from models.SEIR import SEIR
from prepare_data import load_data_germany

if __name__ == "__main__":

    N = 83200000.   # Einwohnerzahl von Deutschland 2019/2020
    r0 = 2.4        # Ansteckungsrate
    gamma = 1 / 3   # Mittlere Infektiöse Zeit
    beta = r0 * gamma
    a = 1 / 5.5     # 1/a Latenzzeit

    # scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True,
    #                         bounds=(-inf, inf), method=None, jac=None, **kwargs)[source]¶

    data = load_data_germany()

    T_data = np.array(data["T"])
    Y_data = np.array(data["cases"]) / N
    print(T_data)
    print(Y_data)
    #exit(0)
    T_min, T_max = min(T_data), max(T_data)
    # T_min, T_max = data["T"].min(), 25
    #T_min, T_max = 25, data["T"].max()
    IR0 = min(data.cases[T_min:T_max])
    E0, I0, R0 = 0., (1. / 3.) * IR0 / N, (2. / 3.) * IR0 / N
    y0 = [1. - E0 - I0 - R0, E0, I0, R0]
    print("IR0", IR0)


    def create_model(_r0):
        return SEIR(_r0 * gamma, gamma, a)


    def fit_func(t, _r0):
        model = create_model(_r0)
        T, Y = model.integrate(y0, min(t), max(t))
        S, E, I, R = model.unravel(Y)
        IpR = [i + r for i, r in zip(I, R)]
        interpolation = scipy.interpolate.interp1d(T, IpR)
        print([interpolation(ti) for ti in t])
        return [interpolation(ti) for ti in t]

    def obj(_r0):
        Y_fitted = fit_func(T_data, _r0)
        return np.sum(np.abs(Y_data - np.array(Y_fitted)))

    res = scipy.optimize.minimize(obj, [r0])

    p0 = [r0]
    popt, pcov = scipy.optimize.curve_fit(
        fit_func,
        xdata= T_data,
        ydata= Y_data,
        p0=p0)
    print(popt)
    best_model = create_model(*popt)
    T, Y = best_model.integrate(y0, T_min, data["T"].max())
    S, E, I, R = SEIR.unravel(Y)

    fig, ax = pyplot.subplots()
    ax.plot(T, [s*N for s in S], "r")
    ax.plot(T, [i*N for i in I], "y")
    ax.plot(T, [r*N for r in R], "b")
    ax.plot(T, [(i+r)*N for i, r in zip(I, R)], "k")
    ax.plot(data["T"], data.cases, "ko")
    ax.set_ylim(0, data.cases.max())
    ax.ticklabel_format(useOffset=False, style='plain')
    pyplot.show()
