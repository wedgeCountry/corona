# coding=utf-8
from typing import Any, Callable

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
    indices = [i for i, b in enumerate(boolean_array) if b]
    return indices


class ModelFitter(object):

    def __init__(self, t, I, R):
        """

        :param t: Zeitachse. Muss sortiert sein!
        :param I: Anzhal momentan infizierter Fälle
        :param R: Anzahl momentan erholter Fälle
        """
        super().__init__()
        mean_infectious_time = 6  # days
        self.gamma = 1 / mean_infectious_time
        self.r0 = 1.5

        self.t = np.array(t)
        self.I = np.array(I)
        self.R = np.array(R)

    def integrate(self, t_min, t_max, y0, x):
        r, g = x, self.gamma
        mdl = SIR(r * g, g)
        return mdl.integrate(y0, t_min, t_max)

    def obj(self, t_min, t_max, t, y, y0, x):
        """

        :param t: Time values for fit
        :param y: Values for fit
        :param y0: Start value of integration
        :param r0: Start value for model
        :return: closeness measure for the resulting values
        """
        T, (S, I, R) = self.integrate(t_min, t_max, y0, x)
        interpolation_function = scipy.interpolate.interp1d(T, I + R)
        y_fitted = np.array([interpolation_function(ti) for ti in t])
        return np.max(np.abs(y - y_fitted))

    def fit(self, from_day, to_day):
        """
        :param from_day: First day of the fitting period
        :param to_day: Last day of the fitting period
        :param r0: Start value for parameter search
        :return: integration function of the best fitted model
        """

        index_from = min(which(self.t >= from_day))
        index_to = max(which(self.t <= to_day))

        R0 = self.R[index_from]
        I0 = self.I[index_from]
        y0 = [1. - I0 - R0, I0, R0]

        t = self.t[index_from: index_to]
        y = self.I[index_from: index_to] + self.R[index_from: index_to]

        def objective(x):
            return self.obj(min(t), max(t), t, y, y0, x)

        best_model = scipy.optimize.minimize(objective, [self.r0])

        print(from_day, to_day, best_model.x)
        print(best_model)

        return best_model.x, lambda up_to_day: self.integrate(from_day, up_to_day, y0, best_model.x)
