import json

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from numpy.polynomial.polynomial import polyval
from scipy.optimize import least_squares
from tqdm import tqdm

from .math import vander_constraint


class DataGenerator(object):
    """
    Creates random function value and finds common tangents. The results serve as training and testing data for a neural
    network which learns to find common tangents between to functions.
    """
    def __init__(self):
        super(DataGenerator, self).__init__()

        self.data_dict = {}

    def generate_data(self, nr_funcs, filepath):
        """
        Randomly generates pairs of functions which have either 0, 1 or 2 common tangents.

        Parameters
        ----------
        nr_funcs : int
            number of functions to generate
        filepath : str
            path to store the resulting json file at

        Returns
        -------

        """

        for i in tqdm(range(nr_funcs)):
            # Get random PolyGenerators
            pg, qg = self.new_funcs_pair()

            # Get the points where the common tangent condition is fulfilled
            pts = self.find_common_tangents(pg, qg)

            # Save the polynomial and the points to a dict
            poly_data = {'p': list(pg.c.round(decimals=4)), 'q': list(qg.c.round(decimals=4)), 'pts': pts}
            self.data_dict[i] = poly_data

        self.save_data_dict(filepath)

    @staticmethod
    def new_funcs_pair():
        # Random degrees of the polynomials
        d_p, d_q = np.random.randint(2, 5, (2, ))

        # PolyGenerators for the two polynomials
        pg = PolyGenerator(d_p)
        qg = PolyGenerator(d_q)

        # Add random constraints (only 0-th derivative) for the polynomial p
        x_p = np.arange(1/(d_p + 2), 1, step=1/(d_p + 2))[:d_p + 1]
        w_p = np.random.uniform(-1., 1., size=(d_p + 1, )) * 1e5

        for x_p_i, w_p_i in zip(x_p, w_p):
            pg.add_constraint(np.array([x_p_i]), 0, w_p_i)

        # dd random constraints (only 0-th derivative) for the polynomial q
        x_q = np.arange(1 / (d_q + 2), 1, step=1 / (d_q + 2))[:d_q + 1]
        w_q = np.random.uniform(-1., 1., size=(d_q + 1,)) * 1e5
        for x_q_i, w_q_i in zip(x_q, w_q):
            qg.add_constraint(np.array([x_q_i]), 0, w_q_i)

        return pg, qg

    @staticmethod
    def find_common_tangents(pg, qg):
        """
        Finds the common tangents between two polynomials by optimization.

        Parameters
        ----------
        pg : PolyGenerator
            polynomial generator for polynomial p
        qg : PolyGenerator
            polynomial generator for polynomial q

        Returns
        -------
        list :
            list of points where the common tangent condition is fulfilled

        """

        # Define the functions
        p = lambda x: pg.get_value_at_x(np.array([x]), 0)
        dp = lambda x: pg.get_value_at_x(np.array([x]), 1)
        q = lambda x: qg.get_value_at_x(np.array([x]), 0)
        dq = lambda x: qg.get_value_at_x(np.array([x]), 1)

        # Pre-allocate storage for the tangent points
        pts = []

        # Define the common tangent condition equations
        def eqns(x):
            x1, x2 = x[0], x[1]
            eq1 = (dp(x1) - dq(x2))[0]
            eq2 = (dp(x1) * (x1 - x2) - (p(x1) - q(x2)))[0]
            return [eq1, eq2]

        lb = (0, 0)  # lower bounds on x1, x2
        ub = (1, 1)  # upper bounds

        # Solve the problem by optimization with varying starting points so that all points are found
        x0s = np.arange(0., 1.1, step=0.1)
        for x0i in x0s:
            for x0j in x0s:
                res = least_squares(eqns, [x0i, x0j], bounds=(lb, ub))
                x_res_1, x_res2 = res.x.round(decimals=4)
                if res.cost < 1e-10 and (x_res_1, x_res2) not in pts:
                    pts.append((x_res_1, x_res2))

        return pts

    def save_data_dict(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.data_dict, outfile)


class PolyGenerator(object):
    """
    Given a set of constraints, PolyGenerator generates a polynomial from the given constraints.

    Parameters
    ----------
    d : int
        polynomial degree
    step_size : int
        step size of the range in which the function is evaluated

    Examples
    --------
    pg = PolyGenerator(2)

    pg.add_constraint(np.array([0.5]), 0, 0)
    pg.add_constraint(np.array([0.5]), 1, 0)
    pg.add_constraint(np.array([0.2]), 0, 2)

    q = pg.get_values()

    """

    def __init__(self, d: int, step_size: int = 1):
        super(PolyGenerator, self).__init__()

        self.d = d

        # Placeholder for constraint vandermonde matrix and and constraint vector
        self.vc = np.empty(shape=(0, self.d + 1))
        self.w = np.empty(shape=(0, ))

        # Placeholder for the coefficients
        self.c = None

        # x-range
        self.x = np.arange(0, 101, step=step_size)/100

    def add_constraint(self, x, o, w):
        """
        Adds a constraint to the constraint vandermonde matrix and vector

        Parameters
        ----------
        x : np.array
            x value where the constraint is located
        o : int
            order of the derivative of the constraint
        w : float
            constraint

        Returns
        -------

        Examples
        --------
        y(1.5) = 0.3 => x=1.5, o=0, w=0.3
        y'(0) = 1 => x=0, o=1, w=1
        y''(2) = 3 => x=2, o=2, w=3

        """

        # Update the constraint vector
        self.w = np.concatenate((self.w, np.array([w])), 0)

        # Update the constraint vandermonde matrix
        v = vander_constraint(x, o, self.d)
        self.vc = np.concatenate((self.vc, v), 0)

    def get_cfs(self):
        """
        For the given constraints, solve the equation system given by self.vc and self.w for the polynomial
        coefficients.

        Equation system:
        w = V*c
        with:
            w: self.w
            V: self.vc
            c: coefficients

        Returns
        -------

        """

        # Check if there are enough constraints for the given degree
        assert self.vc.shape[0] >= self.d + 1

        # Solve the equation system
        self.c = np.flip(pinv(self.vc) @ self.w)

    def get_values(self):
        """
        Given the coefficients, return the function values

        Returns
        -------
        np.array :
            Function values at self.x

        """

        # Check if coefficients are already defined
        if self.c is None:
            self.get_cfs()

        return polyval(self.x, self.c)

    def get_value_at_x(self, x, o):
        """
        Returns the function value at x given the derivative order o.

        Parameters
        ----------
        x : np.array
            x value where the function should be evaluated
        o : int
            derivative order

        Returns
        -------
        float :
            function value at x
        """

        if self.c is None:
            self.get_cfs()

        v = vander_constraint(x, o, self.d)

        return v @ np.flip(self.c)
