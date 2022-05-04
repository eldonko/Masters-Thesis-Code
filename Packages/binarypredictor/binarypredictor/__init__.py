import pkg_resources

import matplotlib.pyplot as plt
import numpy as np
import torch

from .net import DerivativeNet


class BinaryPredictor(object):
    def __init__(self, x_steps=500, net_1='net_1.pth', net_2='net_2.pth'):
        super(BinaryPredictor, self).__init__()

        self.x = torch.arange(1e-10, 1., step=1/x_steps)  # x range

        # Load networks
        stream = pkg_resources.resource_stream(__name__, './models/' + net_1)
        self.net_1 = torch.load(stream)
        stream = pkg_resources.resource_stream(__name__, './models/' + net_2)
        self.net_2 = torch.load(stream)

    @torch.no_grad()
    def predict(self, net_1, net_2, f, g, df, dg, t, scale=1., plot=False, acc=4, threshold=0.3, k=15):
        """
        Predicts the equilibrium compositions of a binary system

        Parameters
        ----------
        net_1 : TangentNet
            network to predict equation 1
        net_2 : TangentNet
            network to predict equation 2
        f : callable
            function for f
        g : callable
            function for g
        df : torch.tensor
            first derivative values of f at x
        dg : torch.tensor
            first derivative values of g at x
        t : float
            temperature at which the equilibrium is to be found
        scale : float
            scaling factor so that the maximum function value is 1
        plot : bool
            whether to plot the results
        acc : int
            accuracy of output values (number of decimals)
        threshold : float
            threshold for the deviation from the tangent's slope and the functions' slopes
        k : int
            select topk results that meet the tangent condition in order to speed up the algorithm
        """
        net_1.eval(), net_2.eval()

        x = self.x

        # Network input
        f_, g_ = f(x, t) / scale, g(x, t) / scale
        inp = torch.hstack((f_, g_))

        # Network outputs
        out_1 = net_1(inp)
        out_2 = net_2(inp)

        # Get the equilibrium compositions by calculating the points of intersections
        # (by approximating as the intersection
        # of the lines connecting the values of out_1 and out_2 at sign changes)
        out_diff = out_1 - out_2
        idx = torch.where(abs(out_diff) < 0.1)[0][:-1]

        if len(idx) == 0:
            return torch.tensor([]), torch.tensor([])

        x_f = torch.hstack((out_1[idx], out_2[idx], out_1[idx + 1], out_2[idx + 1],
                            (out_1[idx] + out_2[idx]) / 2, (out_1[idx + 1] + out_2[idx + 1]) / 2))
        x_g = torch.hstack((x[idx], x[idx], x[idx + 1], x[idx + 1], x[idx], x[idx + 1]))

        # Get the function values at the equilibria
        y_f, y_g = f(x_f, t) / scale, g(x_g, t) / scale

        # Get the slopes of the lines between the equilibria points
        slopes = (y_g - y_f) / (x_g - x_f)

        # Remove lines that are not tangents
        slope_cond = (abs(slopes - dg(x_g, t) / scale) <= threshold) & (abs(slopes - df(x_f, t) / scale) <= threshold)
        idx = torch.where(slope_cond)[0]

        # Recalculate x and y values for all points that are tangent points
        x_f, x_g = x_f[idx], x_g[idx]
        slope_dist = torch.sqrt((slopes[idx] - dg(x_g, t) / scale) ** 2 + (slopes[idx] - df(x_f, t) / scale) ** 2)

        #print(slope_dist)
        # Only take the k best tangents to save time
        idx = torch.topk(slope_dist, min(k, len(slope_dist)), largest=False)[1]
        x_f, x_g = x_f[idx], x_g[idx]
        y_f, y_g = f(x_f, t) / scale, g(x_g, t) / scale
        slope_dist = torch.sqrt((slopes[idx] - dg(x_g, t) / scale) ** 2 + (slopes[idx] - df(x_f, t) / scale) ** 2)
        #print(idx)
        #print(slope_dist)

        # Choose the best tangent if there are multiple results for the same tangent
        if len(x_f) > 0:
            x_eqs = torch.tensor(list(zip(x_f, x_g)))
            s_idx = torch.where(abs(torch.cdist(x_eqs, x_eqs)) < 0.05)

            left, right = s_idx[0], s_idx[1]
            left_unique = torch.unique(left)

            cis = []
            for i in left_unique:
                idx = torch.where(left == i)[0]
                add = right[idx]
                if len(add) > 0:
                    cis.append(add[torch.argmin(slope_dist[add])])
                else:
                    continue
                right = torch.tensor([r for r in right if r not in add])
                left = torch.tensor([l for l in left if l not in add])

            cis = torch.tensor(cis)

            #print(x_f[cis])
            x_f, x_g = torch.unique(x_f[cis]), torch.unique(x_g[cis])
            y_f, y_g = f(x_f, t) / scale, g(x_g, t) / scale

            #print(cis)
            #print(slope_dist[cis])

        # Plot the outputs
        if plot:
            plt.scatter(x, out_1.detach(), s=0.2)
            plt.scatter(x, out_2.detach(), s=0.2)
            plt.title('Network outputs')
            plt.xlabel('x\'\'')
            plt.ylabel('x\'')
            plt.grid()
            plt.show()

            plt.scatter(x, out_diff.detach(), s=0.2)
            plt.title('Difference between network outputs')
            plt.xlabel('x\'\'')
            plt.ylabel('Difference')
            plt.grid()
            plt.show()

            print('x\'_eq: ', np.round(x_f.tolist(), decimals=acc))
            print('x\'\'_eq: ', np.round(x_g.tolist(), decimals=acc))

            for x_f_eq, x_g_eq, y_f_eq, y_g_eq in zip(x_f, x_g, y_f, y_g):
                plt.plot([x_f_eq, x_g_eq], [y_f_eq, y_g_eq], 'ro-')
                plt.plot(x, f_)
                plt.plot(x, g_)
                plt.show()

        return x_f, x_g

    def get_phase_diagram(self, temp_range, f, g, df, dg, plot=False, threshold=0.3, k=15):
        """
        Evaluates the phase diagram for two given functions f & g in a given temperature range

        Parameters
        ----------
        temp_range : range
            temperature range
        f : callable
            function of x (composition) & t (temperature)
        g : callable
            function of x (composition) & t (temperature)
        df : callable
            first derivative of f as function of x (composition) & t (temperature)
        dg : callable
            first derivative of f as function of x (composition) & t (temperature)
        plot : bool
            flag if output is plotted or not
        threshold : float
            threshold for the deviation from the tangent's slope and the functions' slopes
        k : int
            select topk results that meet the tangent condition in order to speed up the algorithm

        Returns
        -------

        xf_eq : torch.tensor
            equilibrium composition on function f for all t in temp_range
        xg_eq : torch.tensor
            equilibrium composition on function g for all t in temp_range
        ts : torch.tensor
            temperatures where a tangent has been found

        """

        x = self.x

        xf_eq, xg_eq, ts = [], [], []

        # Make the predictions
        for t in temp_range:
            scale = max(torch.max(abs(f(x, t))), torch.max(abs(g(x, t))))

            x_f, x_g = self.predict(self.net_1, self.net_2, f, g, df, dg, t,
                                    scale=scale, plot=plot, threshold=threshold, k=k)

            xf_eq.append(x_f) if len(x_f) > 0 else None
            xg_eq.append(x_g) if len(x_g) > 0 else None
            ts.append(t) if len(x_f) > 0 else None

        return xf_eq, xg_eq, ts