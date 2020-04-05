

import scipy.integrate

class EpidemologicalModel:

    def f(self, t, x):
        raise NotImplementedError("This is an abstract method!")

    def integrate(self, y0, T0, Tmax):
        solver = scipy.integrate.RK23(fun=self.f, t0=T0, y0=y0, t_bound=Tmax)

        T, Y = list(), list()
        while solver.t < Tmax:
            T.append(solver.t)
            Y.append(list(solver.y))
            solver.step()
        T.append(solver.t)
        Y.append(list(solver.y))
        return T, Y

    @staticmethod
    def unravel(Y):
        """
        Unravel the integration result into individual variables
        :param Y: integration result
        :return: a list of variables
        """
        raise NotImplementedError("This is an abstract function!")
