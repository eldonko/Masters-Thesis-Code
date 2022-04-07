from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
import torch


def vander_(x, N, increasing=False):
    """
    Creates a vandermonde matrix. Unlike torch.vander, it doesn't make use of inplace operations. This is necessary
    although so that the vandermonde matrix can be used in backward calls.

    Parameters
    ----------
    x : torch.tensor
        input
    N :
        order of the polynomial desired + 1
    increasing : bool
        whether the vandermonde matrix is sorted increasingly or not

    Returns
    -------

    """
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)

    # Initialize the vandermonde matrix by the ones vector in the last (or first) column
    v = torch.ones_like(x)

    # Create the vandermonde matrix
    for i in range(1, N, 1):
        if not increasing:
            v = torch.hstack((x ** i, v))
        else:
            v = torch.hstack((v, x ** i))

    return v


def vander_constraint(x, o, d):
    """
    Creates the constraint vandermonde matrix depending on the derivative order.
    ----------
    x : torch.tensor
        x value of the constraint
    o : int
        order of the constraint (i.e., the order of the derivative)
    d : int
        polynomial degree

    Returns
    -------
    1xd torch.tensor :
        constraint vandermonde matrix

    """

    # Get the vandermonde matrix for x
    v = vander_(x, N=d+1)

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
    dxd torch.tensor
        differentiation matrix

    """
    # Generate the descending range from d-1 to 1 and make a matrix out of it where this range is located on the
    # off-diagonal one entry below the diagonal. In case the derivative order is 0, return the identity matrix
    if o == 0:
        return torch.eye(d + 1)
    else:
        return torch.diag(torch.arange(d, 0., step=-1), -1) ** o


class Polynomial(object):
    """
    Defines a polynomial given a set of coordinates or fits a polynomial (i.e., coefficients) to function values
    """

    def __init__(self, cfs=None):
        """

        Parameters
        ----------
        cfs : torch.tensor
            Polynomial coefficients
        """
        super(Polynomial, self).__init__()

        self.c = cfs  # Polynomial coefficients
        self.d = 0 if cfs is None else len(cfs) - 1  # Polynomial degree
        self.derivative_poly = None  # Derivative polynomial

        # Absolute extreme values inside a range give range specified when evaluating the polynomial
        self.abs_max = 0
        self.abs_min = 0

    def set_cfs(self, c):
        """
        Sets the polynomials coefficients.

        Parameters
        ----------
        c : torch.tensor
            new polynomial coefficients, ordered ascendingly by exponent. For a polynomial p = Ax^2 + Bx + C, the
            vector c should like as follows: c = [A, B, C]

        Returns
        -------

        """

        # Set coefficients
        self.c = c

        # Set derivative poly
        self.set_derivative_poly()

        # Set polynomial degree
        self.d = len(c) - 1

    def get_cfs(self):
        c = self.c.detach().clone()
        return c

    def get_deg(self):
        return self.d

    def from_values(self, x, y, deg=4):
        """
        Approximates a polynomial from a given set of y values over x.

        Parameters
        ----------
        x : torch.tensor
            x range
            shape: [nx1] or [n]
        y : torch.tensor
            y values to approximate as polynomial
            shape: [nx1] or [n]
        deg : polynomial degree

        Returns
        -------

        """

        self.c = (torch.linalg.pinv(vander_(x, deg+1)) @ y).squeeze()
        self.d = deg

    def get_values(self, x, o=0):
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

        if isinstance(x, float):
            x = torch.tensor([x])

        v = vander_constraint(x, o, self.d)

        return v @ self.c

    def set_derivative_poly(self):
        """
        Sets the first derivative of the polynomial as a polynomial itself

        Returns
        -------

        """

        d_c = (self.c * torch.flip(torch.arange(len(self.c)), dims=(-1,)))[:-1]

        # Get the derivative polynomial
        d_poly = Polynomial(d_c)

        self.derivative_poly = d_poly

    def get_derivative_poly(self):
        """
        Returns the first derivative polynomial

        Returns
        -------
        Polynomial :
            first derivative as a polynomial

        """

        return self.derivative_poly

    def max_in_range(self, x: torch.tensor):
        """
        Evaluates the absolute extreme values inside the given range

        Parameters
        ----------
        x : torch.tensor
            range

        Returns
        -------
        float, float :
            absolute minimum, absolute maximum

        """

        vals = self.get_values(x)

        self.abs_min, self.abs_max = abs(torch.min(vals)), abs(torch.max(vals))

        return max(self.abs_min, self.abs_max)

    def scale(self, factor):
        """
        Scales the polynomial by a factor and resets the coefficients accordingly.

        Parameters
        ----------
        factor : float
            scaling factor

        Returns
        -------

        """

        x = torch.arange(-1., 1., step=0.01)
        vals = self.get_values(x)/factor
        self.from_values(x, vals, deg=self.d)

    def plot(self, x: torch.tensor, scatter=False):
        """
        Plots the polynomial over x

        Parameters
        ----------
        x : torch.tensor
            range where the function is evaluated
        scatter : bool
            whether the plot should be scatter or not

        Returns
        -------

        """

        if not scatter:
            plt.plot(x.detach().numpy(), self.get_values(x).detach().numpy())
        else:
            plt.scatter(x.detach().numpy(), self.get_values(x).detach().numpy(), s=.05)

        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.grid()


class PolynomialSet(object):
    """
    A set of polynomials
    """

    def __init__(self):
        super(PolynomialSet, self).__init__()

        self.polys = []  # Store the polynomials as a list

    def __len__(self):
        return len(self.polys)

    def append(self, poly):
        """
        Adds a new polynomial to the polynomial set.

        Parameters
        ----------
        poly : Polynomial
            polynomial to add

        """

        self.polys.append(poly)

    def remove(self, poly):
        """
        Removes a polynomial from the polynomial set.

        Parameters
        ----------
        poly : Polynomial
            polynomial to remove

        """

        self.polys.remove(poly)

    def plot(self, x: torch.tensor, scatter=False):
        """
        Plots all polynomials in the specified x range

        Parameters
        ----------
        x : torch.tensor
            x-range
            either size [n, ] or [n, len(self.polys)]. In the first case, all polynomials are evaluated
            in the same x-range, in the second case, each polynomial can receive and individual x-range.
        scatter : bool
            whether or not to plot as a scatter plot

        """

        if len(x.shape) == 1:
            for poly in self.polys:
                poly.plot(x, scatter=scatter)
        else:
            assert x.shape[-1] == len(self.polys)
            for i in range(len(self.polys)):
                self.polys[i].plot(x[:, i], scatter=scatter)

    def scale_by_max(self, x: torch.tensor):
        """
        Scales the polynomials by the maximum absolute value encountered in all polynomials in the specified range.

        Parameters
        ----------
        x : torch.tensor
            x-range

        """

        max_val = 0

        # Get maximum value
        for poly in self.polys:
            max_val = poly.max_in_range(x) if poly.max_in_range(x) > max_val else max_val

        # Scale all polynomials
        for poly in self.polys:
            poly.scale(max_val)

    def get_derivatives(self):
        """
        Evaluates the derivatives of the polynomials and returns them as a new PolynomialSet

        Returns
        -------
        PolynomialSet :
            set of derivative polynomials

        """

        d_polys = PolynomialSet()

        for poly in self.polys:
            poly.set_derivative_poly()
            d_polys.append(poly.get_derivative_poly())

        return d_polys

    def get_values(self, x: torch.tensor, stack='h', o=0):
        """
        Evaluates the polynomial in the specified range

        Parameters
        ----------
        x : torch.tensor :
            x-range
            either size [n, ] or [n, len(self.polys)]. In the first case, all polynomials are evaluated
            in the same x-range, in the second case, each polynomial can receive and individual x-range.
        stack : str
            whether to stack the values horizontally (h) or vertically (v)
        o : int
            derivative order

        Returns
        -------
        torch.tensor :
            stacked values of the polynomials in the specified x-range.
            either size  [len(self.polys), len(x)] or [len(self.polys) * len(x), ]

        """

        # Input checking
        assert stack in ['h', 'v']

        # Evaluate the polynomials
        vals = []
        if len(x.shape) == 1:
            for poly in self.polys:
                vals.append(poly.get_values(x, o=o).unsqueeze(-1))
        elif len(x.shape) == 2:
            assert x.shape[-1] == len(self.polys)
            for i in range(len(self.polys)):
                vals.append(self.polys[i].get_values(x[:, i], o=o).unsqueeze(-1))
        else:
            assert x.shape[-1] == len(self.polys)
            for i in range(x.shape[-1]):
                print(x[:, :, i].shape)
                vals.append(self.polys[i].get_values(x[:, :, i], o=o).unsqueeze(-1))

        # Stack and return the values
        if stack == 'h':
            return torch.hstack(vals)
        else:
            return torch.vstack(vals)

    def from_values(self, x: torch.tensor, y: torch.tensor):
        """
        Approximates a polynomial set from given values

        Parameters
        ----------
        x : torch.tensor
            x-range
            shape : either [n, ] or [n, k]
        y : torch.tensor
            y-values
            shape : [n, k]

        """

        for i in range(y.shape[1]):
            # Create new polynomial
            new_poly = Polynomial()

            # Approximate values
            if len(x.shape) == 1:
                new_poly.from_values(x, y[:, i])
            else:
                new_poly.from_values(x[:, i], y[:, i])

            # Add to the set
            self.append(new_poly)

    def diff(self, x: torch.tensor, indices):
        """
        Calculates the difference between all polynomials for a given x range which can be different for every
        polynomial

        Parameters
        ----------
        x : torch.tensor
            x-range
            either size [n, ] or [n, 2]. In the first case, all polynomials are evaluated
            in the same x-range, in the second case, each polynomial can receive an individual x-range.
        indices : list
            list of length 2 with the indices of the polynomials to evaluate.

        Returns
        -------
        torch.tensor :
            difference between the polynomial values

        """

        if len(x.shape) == 1:
            return self.polys[indices[0]].get_values(x) - self.polys[indices[1]].get_values(x)
        else:
            return self.polys[indices[0]].get_values(x[:, 0]) - self.polys[indices[1]].get_values(x[:, 1])


class PolynomialBatch(object):
    """
    A mini-batch of PolynomialSets to simplify batched learning
    """

    def __init__(self, batch=None):
        super(PolynomialBatch, self).__init__()

        # PolynomialSets that make up the batch
        self.batch = batch or []

    def set_batch(self, batch):
        """
        Sets the batch

        Parameters
        ----------
        batch : list
            list of PolynomialSet

        """

        self.batch = batch

    def get_values(self, x: torch.tensor, stack='h', o=0):
        """
        Returns the values of all sets in the batch in a given x range

        Parameters
        ----------
        x : torch.tensor
            x-range of length n
        stack : str
            whether to stack the values horizontally (h) or vertically (v)
        o : int
            derivative order

        Returns
        -------
        torch.tensor
            values of all the polynomial sets in the x-range
            shape : [len(self.batch), n, len(PolynomialSet)] or [n, ]

        """

        # Input checking
        assert stack in ['h', 'v']

        vals = []

        if len(x.shape) == 1:
            for i, poly_set in enumerate(self.batch):
                vals.append(poly_set.get_values(x, stack=stack, o=o).unsqueeze(0))
        elif len(x.shape) == 3:
            for i, poly_set in enumerate(self.batch):
                vals.append(poly_set.get_values(x[i], stack=stack, o=o).unsqueeze(0))

        return torch.vstack(vals)

    def get_derivative_batch(self):
        """
        Get the derivatives of all the PolynomialBatch and returns it as a PolynomialBatch itself

        Returns
        -------
        PolynomialBatch :
            Set of polynomials which are the first derivative of all the polynomial sets in this batch

        """

        d_batch = []

        # Get the derivatives of all PolynomialSets in the batch and append them to a list
        for poly_set in self.batch:
            d_batch.append(poly_set.get_derivative())

        # Create a new PolynomialBatch and add the batch to it
        d_sets = PolynomialBatch()
        d_sets.set_batch(d_batch)

        return d_sets


class PolynomialDataLoader(object):
    """
    Class to load polynomials in batches in the form of PolynomialSets.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        super(PolynomialDataLoader, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Maximum number of iterations
        self.max = len(dataset) // batch_size

        # Batched data
        self.batches = []

        # Create batches
        self.create_batches()

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            result = self.batches[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def create_batches(self):
        """
        Orders the data into batches, either shuffled or unshuffled

        Returns
        -------

        """

        indices = np.arange(len(self.dataset))

        # Shuffle indices and dataset if shuffle is True
        if self.shuffle:
            indices = np.random.permutation(indices)
            self.dataset = [self.dataset[i] for i in indices]

        # Arrange the dataset in batches of length batch_size (last one might be shorter depending on whether batch_size
        # divides the length of the dataset or not)
        self.batches = [PolynomialBatch(self.dataset[i:i+self.batch_size]) for i in range(0, len(self.dataset), self.batch_size)]