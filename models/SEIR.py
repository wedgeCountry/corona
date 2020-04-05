# coding=utf-8
import numpy as np

from models.EpidemologicalModel import EpidemologicalModel


class SEIR(EpidemologicalModel):
    """
        https://de.wikipedia.org/wiki/SEIR-Modell

        Bez.    Einheit Beschreibung
        S(t)	1	Anteil der Anfälligen, engl. susceptible. Noch nicht infiziert.
        E(t)	1	Anteil der Exponierten, engl. exposed. Infiziert, aber noch nicht infektiös.
        I(t)	1	Anteil der Infektiösen, engl. infectious.
        R(t)	1	Anteil der Erholten, engl. recovered oder resistant. Bzw. verstorben oder nach Symptomen in Quarantäne.
        t	    d	Zeit in Tagen, engl. time.
        β	   1/d	Transmissionsrate. Der Kehrwert ist die mittlere Zeit zwischen Kontakten.
        γ	   1/d	Gesundungsrate. Der Kehrwert ist die mittlere infektiöse Zeit.
        a	   1/d	Der Kehrwert ist die mittlere Latenzzeit.
    """
    def __init__(self, beta, gamma, a):
        self.beta = beta
        self.gamma = gamma
        self.a = a

    def f(self, t, y):
        S, E, I, R = y
        return np.array([
            -self.beta * S * I,
            self.beta * S * I - self.a * E,
            self.a * E - self.gamma * I,
            self.gamma * I
        ])

    @staticmethod
    def unravel(Y):
        S, E, I, R = [], [], [], []
        for s, e, i, r in Y:
            S.append(s)
            E.append(e)
            I.append(i)
            R.append(r)
        return S, E, I, R
