import numpy as np


def vander_constraint(x, o, d):
    """
    For a polynomial constraint w of the order o at the point x, give the vandermonde constraint matrix.
    Parameters
    ----------
    x : 1x1 np.array
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
    Generates the differentiation matrix M for the vandermonde matrix.

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