"""
This script tests the class DataHandler
"""

# Imports
from DataHandler.Development.DataHandler import DataHandler


def main():
    dh = DataHandler('FactSage'
                     , r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4 Daten\Fact_Fe_100_1000_1.TXT")
    dh.show_table()
    dh.save_table()


if __name__ == "__main__":
    main()
