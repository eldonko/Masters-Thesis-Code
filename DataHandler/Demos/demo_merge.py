"""
This script demonstrates the merge functionality of DataHandler
"""

import matplotlib.pyplot as plt
from DataHandler.Development.DataHandler import DataHandler

first_dh = DataHandler('FactSage'
                       , r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4 Daten\Fact_Fe_100_1000_1.TXT")
second_dh = DataHandler('Dorog'
                        , r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4 Daten\Preprocessed\Fe\Dorog_Fe_10_2000_x.xlsx")

merged_dh = first_dh.merge(second_dh, keep_data_handler=False)

merged_dh.plot_gibbs(temp_col_name='T(K)_bcc')
merged_dh.plot_enthalpy(temp_col_name='T(K)_bcc')

plt.show()
