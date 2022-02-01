import numpy as np
from SGTEHandler.Development.SGTEHandler import SGTEHandler
import matplotlib.pyplot as plt

sgte_handler = SGTEHandler()
sgte_handler.evaluate_equations(200, 6000, 1e5, entropy=True, enthalpy=True, heat_capacity=True)

sgte_handler.get_gibbs_at_temp(300)
sgte_handler.get_gibbs_at_temp(1811)

plt.show()
