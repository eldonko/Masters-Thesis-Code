import torch


def base_function(x=None, t=None, **kwargs):
    r = 8.314

    if x is None:
        x = torch.arange(1e-10, 1., step=kwargs['step'])
    if t is None:
        t = kwargs['T'] if not isinstance(kwargs['T'], torch.Tensor) else kwargs['T'].unsqueeze(-1)
    a = kwargs['a'] if not isinstance(kwargs['a'], torch.Tensor) else kwargs['a'].unsqueeze(-1)
    s_1 = kwargs['s1'] if not isinstance(kwargs['s1'], torch.Tensor) else kwargs['s1'].unsqueeze(-1)
    s_2 = kwargs['s2'] if not isinstance(kwargs['s2'], torch.Tensor) else kwargs['s2'].unsqueeze(-1)
    tm_1 = kwargs['tm1'] if not isinstance(kwargs['tm1'], torch.Tensor) else kwargs['tm1'].unsqueeze(-1)
    tm_2 = kwargs['tm2'] if not isinstance(kwargs['tm2'], torch.Tensor) else kwargs['tm2'].unsqueeze(-1)

    return r * t * (x * torch.log(x) + (1 - x) * torch.log(1 - x)) + a * x * (1 - x) + s_1 * (tm_1 - t) * \
           (1 - x) + s_2 * (tm_2 - t) * x


def first_derivative(x=None, t=None, **kwargs):
    r = 8.314

    if x is None:
        x = torch.arange(1e-10, 1., step=kwargs['step'])
    if t is None:
        t = kwargs['T'] if not isinstance(kwargs['T'], torch.Tensor) else kwargs['T'].unsqueeze(-1)
    a = kwargs['a'] if not isinstance(kwargs['a'], torch.Tensor) else kwargs['a'].unsqueeze(-1)
    s_1 = kwargs['s1'] if not isinstance(kwargs['s1'], torch.Tensor) else kwargs['s1'].unsqueeze(-1)
    s_2 = kwargs['s2'] if not isinstance(kwargs['s2'], torch.Tensor) else kwargs['s2'].unsqueeze(-1)
    tm_1 = kwargs['tm1'] if not isinstance(kwargs['tm1'], torch.Tensor) else kwargs['tm1'].unsqueeze(-1)
    tm_2 = kwargs['tm2'] if not isinstance(kwargs['tm2'], torch.Tensor) else kwargs['tm2'].unsqueeze(-1)

    return r * t * (torch.log(x) - torch.log(1 - x)) + a * (1 - 2 * x) - s_1 * tm_1 + s_2 * tm_2


def second_derivative(x=None, t=None, **kwargs):
    r = 8.314

    if x is None:
        x = torch.arange(1e-10, 1., step=kwargs['step'])
    if t is None:
        t = kwargs['T'] if not isinstance(kwargs['T'], torch.Tensor) else kwargs['T'].unsqueeze(-1)
    a = kwargs['a'] if not isinstance(kwargs['a'], torch.Tensor) else kwargs['T'].unsqueeze(-1)

    return r * t * (1 / x + 1 / (1 - x)) - 2 * a