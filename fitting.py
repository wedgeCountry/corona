# coding=utf-8
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import numpy as np
from matplotlib import pyplot
import datetime
import pandas as pd

from models.ModelFitter import ModelFitter
from models.SEIR import SEIR
from models.SIR import SIR
from prepare_data import load_data_germany
from datetime import date


def which(boolean_array):
    return [i for i, b in enumerate(boolean_array) if b]


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
    first_day_with_1000_infections = days[min(which(confirmed > 1000))]

    N = 83200000.  # Einwohnerzahl von Deutschland 2019/2020

    fitter = ModelFitter(t=days,
                         I=(confirmed - recovered) / N,
                         R=recovered / N)

#    best_1, fit_1 = fitter.fit(first_day_with_1000_infections,
#                       first_day_with_1000_infections + 15)

    best_2, fit_2 = fitter.fit(first_day_with_1000_infections + 15,
                       first_day_with_1000_infections + 15 + 15)

    fig, ax = pyplot.subplots()
    ax.plot(days, confirmed, "ko")
    ax.plot(days, recovered, "bo")

    for fit in [fit_2]:
        T, (S, I, R) = fit(max(days) + 4)

        ax.plot(T, N * I, "y")
        ax.plot(T, N * R, "b")
        ax.plot(T, N * (I + R), "k")

    ax.ticklabel_format(useOffset=False, style='plain')
    pyplot.show()
