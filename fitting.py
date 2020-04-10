# coding=utf-8
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import numpy as np
from matplotlib import pyplot
import datetime
import pandas as pd

from models.SEIR import SEIR
from models.SIR import SIR
from prepare_data import load_data_germany
from datetime import date

def which(boolean_array):
    for i, b in enumerate(boolean_array):
        if b:
            yield i

if __name__ == "__main__":


    data_ger = pd.read_csv("data/Hopkins_timeseries_Germany.csv", sep=";")

    data_ger["date"] = [date(2000 + int(d.split("/")[2]),
          int(d.split("/")[0]),
          int(d.split("/")[1])) for d in data_ger["T"]]
    data_ger["confirmed"] = data_ger["confirmed.V1"]
    data_ger["recovered"] = data_ger["recovered.V1"]
    data_ger["dead"] = data_ger["dead.V1"]
    #exit(0)

    #print(T_data)
    days = np.array([td.days for td in data_ger["date"] - min(data_ger["date"])])
    confirmed = np.array(data_ger["confirmed"])
    recovered = np.array(data_ger["recovered"])
    mindex = min(which(confirmed > 1000))

    confirmed = confirmed[mindex:]
    recovered = recovered[mindex:]
    days = days[mindex:]
    print("days", days)
    print("confirmed", confirmed)

    N = 83200000.  # Einwohnerzahl von Deutschland 2019/2020

    R_data = recovered / N
    I_data = (confirmed - recovered) / N

    day_0 = days[0]
    R0 = R_data[0]
    I0 = I_data[0]
    y0 = [1. - I0 - R0, I0, R0]

    r0 = 1.2        # Ansteckungsrate
    gamma = 1 / 3   # 1/gamma: Mittlere Infektiöse Zeit
    beta = r0 * gamma

    print("y0", y0)

    def create_model(_r0):
        return SIR(_r0 * gamma, gamma)

    def fit_func(t, y0, _r0):
        mdl = create_model(_r0)
        ti, yi = mdl.integrate(y0, min(t), max(t))
        S, I, R = mdl.unravel(yi)
        IpR = [i + r for i, r in zip(I, R)]
        interpolation = scipy.interpolate.interp1d(ti, IpR)
        return [interpolation(i) for i in t]

    y0_from_start = [1. - I0 - R0, I0, R0]

    def obj_from_start(_r0, t_input, y_input):
        y_fitted = fit_func(t_input, y0_from_start, _r0)
        return max(np.abs(y_input - np.array(y_fitted)))

    res_from_start = scipy.optimize.minimize(lambda x: obj_from_start(x, days[1:10], I_data[1:10]), [r0])

    y0_ab_day_65 = [1. - I_data[(65 - day_0)] - R_data[(65 - day_0)], I_data[(65 - day_0)], R_data[(65 - day_0)]]
    print(y0_ab_day_65)

    def obj_ab_day_65(_r0):
        t_input = days[(65 - day_0):]
        y_input = I_data[(65 - day_0):]
        y_fitted = fit_func(t_input, y0_ab_day_65, _r0)
        return sum((y_input - np.array(y_fitted))**2)

    res_ab_day_65 = scipy.optimize.minimize(obj_ab_day_65, [1.2])

    #exit(0)

    to_day = max(days)
    model_from_start = SIR(res_from_start.x * gamma, gamma)
    T_from_start, Y_from_start = model_from_start.integrate(y0_from_start, min(days), to_day)
    S_from_start, I_from_start, R_from_start = model_from_start.unravel(Y_from_start)

    model_ab_day_65 = SIR(res_ab_day_65.x * gamma, gamma)
    T_ab_day_65, Y_ab_day_65 = model_ab_day_65.integrate(y0_ab_day_65, days[(65 - day_0)], to_day)
    S_ab_day_65, I_ab_day_65, R_ab_day_65 = model_ab_day_65.unravel(Y_ab_day_65)

    fig, ax = pyplot.subplots()
    ax.plot(T_from_start, N * I_from_start, "y")
    ax.plot(T_from_start, N * R_from_start, "b")
    ax.plot(T_from_start, N * (I_from_start + R_from_start), "k")
   # ax.plot(T_ab_day_65, N * I_ab_day_65, "y-")
   # ax.plot(T_ab_day_65, N * R_ab_day_65, "b-")
   # ax.plot(T_ab_day_65, N * (I_ab_day_65 + R_ab_day_65), "k-")
    ax.plot(days, N * (I_data + R_data), "ko")
    #ax.set_ylim(0, data_ger.confirmed.max())
    ax.ticklabel_format(useOffset=False, style='plain')
    pyplot.show()

    exit(0)


    p0 = [r0]
    popt, pcov = scipy.optimize.curve_fit(
        fit_func,
        xdata=T_data,
        ydata=Y_data,
        p0=p0)
    print(popt, obj(popt))
    #exit(0)
    best_model = create_model(*popt)
    T, Y = best_model.integrate(y0, T_data[0], Y_data.max())
    S, E, I, R = SEIR.unravel(Y)

    fig, ax = pyplot.subplots()
    ax.plot(T, [s*N for s in S], "r")
    ax.plot(T, [i*N for i in I], "y")
    ax.plot(T, [r*N for r in R], "b")
    ax.plot(T, [(i+r)*N for i, r in zip(I, R)], "k")
    #ax.plot(T_data, data_ger.confirmed, "ko")
    #ax.set_ylim(0, data_ger.confirmed.max())
    ax.ticklabel_format(useOffset=False, style='plain')
    pyplot.show()

