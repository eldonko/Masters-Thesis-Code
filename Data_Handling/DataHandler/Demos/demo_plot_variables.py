"""
This script demonstrates the variable plotting functionality of DataHandler
"""

import matplotlib.pyplot as plt

from DataHandler.Development.DataHandler import DataHandler

dh = DataHandler('Dorog'
                 , r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4 Daten\Preprocessed\Fe\Fact_Fe_100_1000_1.xlsx")
dh.show_table()

dh.plot_gibbs()
dh.plot_enthalpy()
dh.plot_heat_capacity()
dh.plot_entropy()

plt.show()
