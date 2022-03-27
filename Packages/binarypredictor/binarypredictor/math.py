import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
import torch


def vander_constraint(x, o, d):
    """
    Wrapper function for the constraint vandermonde matrix. If x is a numpy array, the constraint matrix will be
    also a numpy array, if it is a torch tensor, it will be a torch tensor.
    ----------
    x : np.array or torch.tensor
        x value of the constraint
    o : int
        order of the constraint (i.e., the order of the derivative)
    d : int
        polynomial degree

    Returns
    -------
    1xd np.array :
        constraint vandermonde matrix

    """

    # Get the vandermonde matrix for x
    v = np.vander(x, N=d+1)

    # Get the differentiation matrix
    m = vander_diff_matrix(d, o)

    # Get and return the constraint vandermonde matrix
    return v @ m


def vander_diff_matrix(d, o):
    """
    Generates the differentiation matrix M for the Vandermonde matrix.

    For the 4x4 case, the differentiation matrix looks as follows:
    0   0   0   0
    3   0   0   0
    0   2   0   0
    0   0   1   0

    Parameters
    ----------
    d : int
        polynomial degree
    o : int
        order of the derivative

    Returns
    -------
    dxd np.array
        differentiation matrix

    """
    # Generate the descending range from d-1 to 1 and make a matrix out of it where this range is located on the
    # off-diagonal one entry below the diagonal. In case the derivative order is 0, return the identity matrix
    if o == 0:
        return np.eye(d + 1)
    else:
        return np.diag(np.arange(d, 0, step=-1), -1) ** o


class Polynomial(object):
    """
    Defines a polynomial given a set of coordinates or fits a polynomial (i.e., coefficients) to function values
    """

    def __init__(self):
        super(Polynomial, self).__init__()

        self.c = None  # Polynomial coefficients
        self.d = 0  # Polynomial degree

    def set_cfs(self, c):
        """
        Sets the polynomials coefficients.

        Parameters
        ----------
        c : np.array, list or torch.tensor
            new polynomial coefficients, ordered ascendingly by exponent. For a polynomial p = Ax^2 + Bx + C, the
            vector c should like as follows: c = [C, B, A]

        Returns
        -------

        """

        self.c = c

        # Set polynomial degree
        self.d = len(c) - 1

    def get_cfs(self):
        return self.c

    def get_values(self, x):
        """
        Given the coefficients, return the function values

        Parameters
        ----------
        x : np.array or torch.tensor
            range where the function is evaluated

        Returns
        -------
        np.array or torch.tensor:
            Function values at self.x

        Examples
        --------

        get_values(np.arange(0., 1., step=0.01))

        """

        # Check if coefficients are already defined
        if self.c is None:
            raise ValueError('Coefficients of polynomial must be defined before evaluating the function.')

        # Evaluate the polynomial based on whether the x-range is a numpy array or torch tensor
        if isinstance(x, np.ndarray):
            values = polyval(x, self.c)
        elif isinstance(x, torch.Tensor):
            if isinstance(self.c, np.ndarray):
                self.c = torch.tensor(self.c, requires_grad=True)

            values = torch.vander(x, N=self.d+1) @ self.c.float()
        else:
            raise TypeError('x must be either np.array or torch.tensor')

        return values

    def get_value_at_x(self, x, o):
        """
        Returns the function value at x given the derivative order o.

        Parameters
        ----------
        x : float
            x value where the function should be evaluated
        o : int
            derivative order

        Returns
        -------
        float :
            function value at x
        """

        if self.c is None:
            raise ValueError('Coefficients of polynomial must be defined before evaluating the function.')

        v = vander_constraint(x, o, self.d)

        return v @ np.flip(self.c)

    def get_derivative_poly(self, x: np.array):
        """
        Returns the first derivative of the polynomial as a polynomial itself

        Parameters
        ----------
        x : np.array
            x-range in which the derivative is evaluated

        Returns
        -------
        Polynomial :
            first derivative as a polynomial

        """

        # Get the derivative values
        p_prime = self.get_value_at_x(x, 1)

        # Get the derivative's coefficients
        c = polyfit(x, p_prime, self.d-1)

        # Get the derivative polynomial
        d_poly = Polynomial()
        d_poly.set_cfs(c)

        return d_poly

    def plot(self, x: np.array):
        """
        Plots the polynomial over x

        Parameters
        ----------
        x : np.array
            range where the function is evaluated

        Returns
        -------

        """

        plt.plot(x, self.get_values(x))
        plt.xlabel('x')
        plt.ylabel('p(x)')