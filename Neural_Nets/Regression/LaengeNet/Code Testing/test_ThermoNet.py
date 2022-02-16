from ThermoNet.Development.ThermoNet import ThermoNet
from SGTE.Development.SGTEHandler import SGTEHandler
import numpy as np
import matplotlib.pyplot as plt


def main():
	thermo_net = ThermoNet(20)
	sgte_handler = SGTEHandler('Fe')

	sgte_handler.evaluate_equations(200, 2000, 1e5, plot=False, phases=['BCC_A2'])

	gibbs_data = sgte_handler.equation_result_data

	data = np.array(gibbs_data['G(J)_BCC_A2'])
	temp = np.array(gibbs_data['Temperature'])

	noise = np.random.normal(0, 1500, size=(len(data), ))
	data = data + noise

	plt.scatter(gibbs_data['Temperature'], data, s=0.2)
	plt.show()

	x = np.random.randn(1, 1)
	g = np.random.randn(1, 1)
	print(x)
	output, cache = thermo_net(x)

	thermo_net.zero_grad()
	thermo_net.backward(output, g, cache)
	thermo_net.update(1e-3)


if __name__ == '__main__':
	main()
