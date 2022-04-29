import torch

from .net import DerivativeNet


def split_functions(first_derivative_net, second_derivative_net, functions):
    """
    Gets the first and second derivative of a function and uses the turning points to split the functions.

    Parameters
    ----------
    first_derivative_net : DerivativeNet
        net to predict the first derivative
    second_derivative_net : DerivativeNet
        net to predict the second derivative
    functions : torch.tensor
        function values
        shape : [k, n, 2] or [n, 2], where k: batch size, n: number of points in the range

    Returns
    -------

    """

    # Separate the functions in the tensor
    f = functions[:, 0]
    g = functions[:, 1]

    # First derivative
    f_ = first_derivative_net(f)
    g_ = first_derivative_net(g)

    # Second derivative
    f__ = second_derivative_net(f_)
    g__ = second_derivative_net(g_)

    return split(f, f__), split(g, g__)


def split(f, f__):
    """
    Does the actual function splitting

    Parameters
    ----------
    f : torch.tensor
        base function
    f__ : torch.tensor
        second derivative of f

    Returns
    -------

    """

    # Get the roots of the second derivative function
    idx = torch.where(torch.diff(torch.sign(f__)) != 0)[0]
    idx = idx[idx != 0]

    # Do the splitting, whereas a minimum distance of 5 between two roots is required for them to be valid
    f_splits = []
    if len(idx) > 0:
        min_diff = 5
        valid_idx = [idx[j] for j in range(1, len(idx) - 1) if
                     idx[j + 1] - idx[j] > min_diff and abs(idx[j] - idx[j - 1]) > min_diff]
        if idx[-1] - idx[0] > min_diff:
            valid_idx += [idx[0]] + [idx[-1]] + [0] + [len(f)]
        valid_idx = torch.tensor(sorted(valid_idx), dtype=torch.int64)

        # Remove curve parts with negative curvature
        for j in range(len(valid_idx) - 1):
            s = f__[valid_idx[j]:valid_idx[j + 1]]
            if torch.sum(torch.sign(s)) >= 0:
                z = torch.zeros_like(f)
                z[valid_idx[j]:valid_idx[j + 1]] = f[valid_idx[j]:valid_idx[j + 1]]
                f_splits.append((z, (valid_idx[j].item(), valid_idx[j + 1].item())))
    else:
        f_splits.append((f, (0, len(f))))

    return f_splits