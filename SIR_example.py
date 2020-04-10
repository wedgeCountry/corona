
# coding=utf-8
import scipy.integrate
import numpy as np
from matplotlib import pyplot

from models.SEIR import SEIR
from models.SIR import SIR

if __name__ == "__main__":

    # data: https://github.com/CSSEGISandData/COVID-19
    data_path = "data/COVID-19-master/csse_covid_19_data/"

    N = 83200000.   # Einwohnerzahl von Deutschland 2019/2020
    r0 = 1.07          # Ansteckungsrate
    gamma = 1 / 4   # Mittlere Infektiöse Zeit
    beta = r0 * gamma

    I0, R0 = 121000.0/N, 42000/N
    y0 = [1. - I0 - R0, I0, R0]
    days = 1400

    model = SIR(beta, gamma)
    T, (S, I, R) = model.integrate(y0, 0, days)

    fig, ax = pyplot.subplots()
    #ax.plot(T, S, "r")
    ax.plot(T, I, "y")
    ax.plot(T, R, "b")
    ax.ticklabel_format(useOffset=False, style='plain')
    pyplot.show()
