import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import torch
from torch.linalg import pinv
from tqdm import tqdm

from .poly import vander_constraint, Polynomial, PolynomialSet


class DataGenerator(object):
    """
    Creates random function value and finds common tangents. The results serve as training and testing data for a neural
    network which learns to find common tangents between to functions.
    """
    def __init__(self):
        super(DataGenerator, self).__init__()

        self.polys = []

    def generate_data(self, nr_polys):
        """
        Randomly generates pairs of polynomials.

        Parameters
        ----------
        nr_polys : int
            number of functions to generate

        """

        for i in tqdm(range(nr_polys)):
            # Get random PolySets
            poly_set = self.new_funcs_pair()

            self.polys.append(poly_set)

    def get_polys(self):
        """
        Returns the obtained polynomial sets.

        Returns
        -------
        list :
            list of PolynomialSets obtained

        """

        return self.polys

    @staticmethod
    def new_funcs_pair():
        # Random degrees of the polynomials
        d_p, d_q = np.random.randint(2, 5, (2, ))

        # PolyGenerators for the two polynomials
        pg = PolyGenerator(d_p)
        qg = PolyGenerator(d_q)

        # Add random constraints (only 0-th derivative) for the polynomial p at fixed x-points
        x_p = torch.arange(1/(d_p + 2), 1, step=1/(d_p + 2))[:d_p + 1]
        w_p = torch.from_numpy(np.random.uniform(-1., 1., size=(d_p + 1, )).astype(np.float32))

        for x_p_i, w_p_i in zip(x_p, w_p):
            pg.add_constraint(x_p_i, 0, w_p_i)

        # Add random constraints (only 0-th derivative) for the polynomial q at fixed x-points
        x_q = torch.arange(1 / (d_q + 2), 1, step=1 / (d_q + 2))[:d_q + 1]
        w_q = torch.from_numpy(np.random.uniform(-1., 1., size=(d_q + 1, )).astype(np.float32))
        for x_q_i, w_q_i in zip(x_q, w_q):
            qg.add_constraint(x_q_i, 0, w_q_i)

        # Create a polynomial set and at the polynomials obtained by the polynomial generators
        poly_set = PolynomialSet()
        poly_set.append(Polynomial(pg.get_cfs()))
        poly_set.append(Polynomial(qg.get_cfs()))

        # Scale the set so that the maximum value in the specified x-range is either 1 or -1
        x = torch.arange(0., 1., step=.01)
        poly_set.scale_by_max(x)

        return poly_set


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

    pg.add_constraint(0.5, 0, 0)
    pg.add_constraint(0.5, 1, 0)
    pg.add_constraint(0.2, 0, 2)

    q = pg.get_values()

    """

    def __init__(self, d: int, step_size: int = 1):
        super(PolyGenerator, self).__init__()

        self.d = d

        # Placeholder for constraint vandermonde matrix and and constraint vector
        self.vc = torch.empty(size=(0, self.d + 1), requires_grad=True)
        self.w = torch.empty(size=(0, ), requires_grad=True)

        # Placeholder for the coefficients
        self.c = None

        # x-range
        self.x = torch.arange(0., 101., step=step_size, requires_grad=True)/100

    def add_constraint(self, x, o, w):
        """
        Adds a constraint to the constraint vandermonde matrix and vector

        Parameters
        ----------
        x : float
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
        self.w = torch.concat((self.w, torch.tensor([w])), 0)

        # Update the constraint vandermonde matrix
        v = vander_constraint(torch.tensor([x]), o, self.d)
        self.vc = torch.concat((self.vc, v), 0)

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
        torch.tensor :
            polynomial coefficients

        """

        # Check if there are enough constraints for the given degree
        assert self.vc.shape[0] >= self.d + 1

        # Solve the equation system
        self.c = pinv(self.vc) @ self.w

        return self.c
