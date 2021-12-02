from ThermoNet.Development.ThermoNet import ThermoNet
import numpy as np


def main():
	thermo_net = ThermoNet(1, 5, 1)
	x = np.random.randn(1, 1)
	g = np.random.randn(1, 1)
	print(x)
	loss, cache = thermo_net(x, g)

	thermo_net.zero_grad()
	thermo_net.backward(loss, cache)


if __name__ == '__main__':
	main()
