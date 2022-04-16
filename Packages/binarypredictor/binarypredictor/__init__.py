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

    if len(functions.shape) < 3:
        functions = functions.unsqueeze(0)

    # Separate the functions in the tensor
    f = functions[:, :, 0]
    g = functions[:, :, 1]

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
    row, idx = torch.where(torch.diff(torch.sign(f__)) != 0)
    row, idx = row[idx != 0], idx[idx != 0]

    # Get the indices to split
    bin_count = torch.cumsum(torch.bincount(row, minlength=f.shape[0]), dim=0)
    idx = torch.tensor_split(idx, bin_count)[:f.shape[0]]

    # Do the splitting, whereas a minimum distance of 5 between two roots is required for them to be valid
    f_splits = []
    for n, i in enumerate(idx):
        if len(i) > 0:
            min_diff = 5
            valid_idx = [i[j] for j in range(1, len(i) - 1) if
                         i[j + 1] - i[j] > min_diff and abs(i[j] - i[j - 1]) > min_diff]
            if i[-1] - i[0] > min_diff:
                valid_idx += [i[0]] + [i[-1]] + [0] + [f.shape[1]]
            valid_idx = torch.tensor(sorted(valid_idx), dtype=torch.int64)

            f_split_ = []
            # Remove curve parts with negative curvature
            for j in range(len(valid_idx) - 1):
                s = f__[j, valid_idx[j]:valid_idx[j + 1]]
                if torch.sum(torch.sign(s)) >= 0:
                    z = torch.zeros(size=(f.shape[1], ))
                    z[valid_idx[j]:valid_idx[j + 1]] = f[n, valid_idx[j]:valid_idx[j + 1]]
                    f_split_.append(z)

            f_splits.append(f_split_)

    return f_splits