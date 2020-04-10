# coding=utf-8
import numpy as np

from models.EpidemologicalModel import EpidemologicalModel


class SIR(EpidemologicalModel):
    """
        https://de.wikipedia.org/wiki/SEIR-Modell

        Bez.    Einheit Beschreibung
        S(t)	1	Anteil der Anfälligen, engl. susceptible. Noch nicht infiziert.
        I(t)	1	Anteil der Infektiösen, engl. infectious.
        R(t)	1	Anteil der Erholten, engl. recovered oder resistant. Bzw. verstorben oder nach Symptomen in Quarantäne.
        t	    d	Zeit in Tagen, engl. time.
        β	   1/d	Transmissionsrate. Der Kehrwert ist die mittlere Zeit zwischen Kontakten.
        γ	   1/d	Gesundungsrate. Der Kehrwert ist die mittlere infektiöse Zeit.
    """
    def __init__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

    def f(self, t, y):
        S, I, R = y
        return np.array([
            -self.beta * S * I,
            self.beta * S * I - self.gamma * I,
            self.gamma * I
        ])

    @staticmethod
    def unravel(Y):
        S, I, R = [], [], []
        for s, i, r in Y:
            S.append(s)
            I.append(i)
            R.append(r)
        return np.array(S), np.array(I), np.array(R)
