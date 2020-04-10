
# coding=utf-8
import scipy.integrate
import numpy as np
from matplotlib import pyplot

from models.SEIR import SEIR

if __name__ == "__main__":

    # data: https://github.com/CSSEGISandData/COVID-19
    data_path = "data/COVID-19-master/csse_covid_19_data/"

    N = 83200000.   # Einwohnerzahl von Deutschland 2019/2020
    r0 = 3          # Ansteckungsrate
    gamma = 1 / 3   # Mittlere Infektiöse Zeit
    beta = r0 * gamma
    a = 1 / 5.5     # 1/a Mittlere Latenzzeit 5.5 Tage

    E0, I0, R0 = 1000.0/N, 100.0/N, 0
    y0 = [1. - E0 - I0 - R0, E0, I0, R0]
    days = 140

    model = SEIR(beta, gamma, a)
    T, Y = model.integrate(y0, 0, days)
    S, E, I, R = model.unravel(Y)

    fig, ax = pyplot.subplots()
    ax.plot(T, S, "r")
    ax.plot(T, I, "y")
    ax.plot(T, R, "b")
    ax.ticklabel_format(useOffset=False, style='plain')
    pyplot.show()
