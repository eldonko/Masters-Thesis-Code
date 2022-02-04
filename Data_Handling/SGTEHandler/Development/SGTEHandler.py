import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
import os
import math


class SGTEHandler(object):
    def __init__(self, element=None):
        super(SGTEHandler, self).__init__()
        self.data = None
        self.colors = ['royalblue', 'seagreen', 'slategrey', 'red', 'pink', 'orange', 'mintcream', 'lime', 'yellow']
        self.element = None
        self.elements = ['Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cs',
                         'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 'In', 'Ir', 'K', 'La', 'Li',
                         'Lu', 'Mg', 'Mn', 'Mo', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'Os', 'P', 'Pa', 'Pb', 'Pd', 'Pr', 'Pt',
                         'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc',
                         'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

        self.errors = {'INP_1': '*** INP_1 *** Invalid input, please try again!'}

        self.equation_result_data = None
        self.selected_phases = []

        self.set_element(element)

    def set_element(self, element=None):
        """
        Allows either user input for element or sets the element to the provided input of element is not None
        :param element: (optional) element for which the equations shall be evaluated
        :return:
        """

        if element is not None and element in self.elements:
            self.element = element

        while self.element not in self.elements:
            self.element = input('Please enter element name: ').capitalize()

        print(self.element + ' successfully selected!\n')

        inp_file = os.path.join(r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4_Daten\SGTE Data", self.element + '.xlsx')
        self.load_data(inp_file)

    def set_phases(self, phases=None):
        """
        Prompts the user to specify the phases he would like to evaluate or sets the phases to the phases specified
        :return:
        """
        # Get all the unique phases in self.data
        possible_phases = self.data['Phase'].unique()

        if phases is not None:
            # TODO: Add error handling
            if phases == ['all']:
                self.selected_phases = possible_phases
            else:
                self.selected_phases = phases
            return

        if len(self.selected_phases) > 0:
            print('To reuse phases from before, enter y. To reset phases, enter n.')
            keep = ''
            while keep not in ['y', 'n']:
                keep = input('[y/n]: ').strip()
                # If input is y, return as nothing has to be changed. If input is n, proceed with phase selection
                if keep == 'y':
                    return

        print('Please select the phases you want to evaluate!')
        print('Possible phases: ')
        for phase in possible_phases:
            print(phase)
        print('\nEnter a phases you want to evaluate. Example: BCC_A2')
        print('To select all phases, enter: all')
        print('If done, press Enter')

        # Get the input for all phases which are to be selected
        selected_phases = []
        done = False
        while not done or len(selected_phases) == 0:
            inp = input('Enter phase: ')
            if inp.strip() == '':
                if len(selected_phases) > 0:
                    done = True
                else:
                    print('Please enter phase before pressing Enter')
            elif inp == 'all':
                selected_phases = possible_phases
                done = True
            elif inp.upper() in possible_phases:
                selected_phases.append(inp.upper())
            else:
                print(self.errors['INP_1'])

        print()
        self.selected_phases = selected_phases

    def evaluate_equations(self, start_temp, end_temp, p, gibbs=True, entropy=False, enthalpy=False, heat_capacity=False
                           , plot=True, phases=None, step=1):
        """
        Evaluates the Gibbs energy for a temperature range and selected phase at a certain pressure.
        :param start_temp: lower bound of the temperature interval
        :param end_temp: upper bound of the temperature interval
        :param p: pressure
        :param gibbs: (optional) if True, Gibbs energy is evaluated
        :param entropy: (optional) if True, entropy is evaluated
        :param enthalpy: (optional) if True, enthalpy is evaluated
        :param heat_capacity: (optional) if True, heat capacity is evaluated
        :param plot: (optional) Determines if data shall be plotted
        :param phases: (optional) phases to be evaluated can be selected
        :param step: step size of temp_range
        :return:
        """

        self.set_phases(phases)

        temp_range = np.arange(start_temp, end_temp, step, dtype=np.float32)
        ax_gibbs = None
        ax_entropy = None
        ax_enthalpy = None
        ax_heat_capacity = None

        for i, phase in enumerate(self.selected_phases):
            # Get the Gibbs energies for the temperature range
            equation_result = self.solve_equations(phase, p, temp_range, gibbs, entropy, enthalpy, heat_capacity)
            # Plot the obtained data
            if plot:
                if gibbs:
                    ax_gibbs = self.plot_data(equation_result, ax_gibbs, 'G(J)_', phase, i)
                if entropy:
                    ax_entropy = self.plot_data(equation_result, ax_entropy, 'S(J/K)_', phase, i)
                if enthalpy:
                    ax_enthalpy = self.plot_data(equation_result, ax_enthalpy, 'H(J)_', phase, i)
                if heat_capacity:
                    ax_heat_capacity = self.plot_data(equation_result, ax_heat_capacity, 'Cp(J/K)_', phase, i)

            # Save the obtained data to the SGTEHandler
            if self.equation_result_data is None:
                self.equation_result_data = equation_result
            else:
                self.equation_result_data = self.equation_result_data.merge(equation_result, on='Temperature')

        if plot:
            if gibbs:
                ax_gibbs.legend()
            if entropy:
                ax_entropy.legend()
            if enthalpy:
                ax_enthalpy.legend()
            if heat_capacity:
                ax_heat_capacity.legend()

    def plot_data(self, equation_result, ax, prefix, phase, i):
        """
        Plots the data of the equation results
        :param equation_result: the result DataFrame
        :param ax: the axis to plot to
        :param prefix: the prefix of the column name
        :param phase: the evaluated phase
        :param i: phase index
        :return:
        """

        if ax is None:
            ax = equation_result.plot(x='Temperature', y=prefix + phase, kind='scatter', grid=True, s=0.5,
                                      color=self.colors[i], label=phase, ylabel=prefix.replace('_', ''))
        else:
            equation_result.plot(x='Temperature', y=prefix + phase, kind='scatter', grid=True, s=0.5, ax=ax,
                                 color=self.colors[i], label=phase, ylabel=prefix.replace('_', ''))

        return ax

    def load_data(self, input_file):
        """
        Loads the data from a specified excel file and fills all NaN values with 0
        :param input_file: Excel file which contains SGTE Data
        :return:
        """
        self.data = pd.read_excel(input_file)
        self.data = self.data.fillna(0)

        # Convert the column names which are integers from str to int
        for col in self.data.columns.values:
            try:
                i_col = int(col)
                self.data = self.data.rename(columns={col: i_col})
            except:
                pass

    def solve_equations(self, phase, p, temp, gibbs, entropy, enthalpy, heat_capacity):
        """
        Evaluates the equations specified by Dinsdale (SGTE). For further reference see "SGTE Data for
        Pure Elements [Dinsdale]"
        :param temp: temperature
        :param phase: The phase in which the Gibbs energy is to be evaluated.
        :param p: pressure
        :param gibbs: if True, Gibbs energy is evaluated
        :param entropy: if True, entropy is evaluated
        :param enthalpy: if True, enthalpy is evaluated
        :param heat_capacity: if True, heat capacity is evaluated
        :return:
        """

        # Get the SGTE coefficients from self.data
        data = self.data[self.data['Phase'] == phase].sort_values(by='Start temperature', ascending=True).reset_index(
            drop=True)

        # Allow for temperature ranges which go over the intervals specified in SGTE data. For this, it is necessary to
        # split the temperature range in order to be able to compute the equations accordingly.
        temp_ranges = self.get_temp_ranges(data, temp)

        # Collection of the obtained data
        temp_ranges_total = np.empty([0])
        gibbs_energies_total = np.empty([0])
        entropies_total = np.empty([0])
        enthalpies_total = np.empty([0])
        heat_capacities_total = np.empty([0])

        # Evaluate the equations by getting the coefficients from data
        for i, temp_range in enumerate(temp_ranges):
            # Store the temperature range
            temp_ranges_total = np.concatenate((temp_ranges_total, temp_range), axis=None)

            # Get the c value
            c = data['c'][i]

            # Get the coefficients of every T that exists in an exponential in the Gibbs equation
            num_cols = [(nums, data[nums][i]) for nums in data.columns.values if type(nums) == int]

            # Get the sum of the polynomial parts in the equations
            d_sum_gibbs = 0
            d_sum_entropies = 0
            d_sum_enthalpies = 0
            d_sum_heat_capacities = 0
            for val in num_cols:
                if gibbs:
                    d_sum_gibbs += val[1] * temp_range ** val[0]
                if entropy:
                    d_sum_entropies -= val[1] * val[0] * temp_range ** (val[0] - 1)
                if enthalpy:
                    d_sum_enthalpies -= val[1] * (val[0] - 1) * temp_range ** val[0]

                    if val[0] == 0:
                        d_sum_enthalpies += val[1] * (val[0] - 1) * temp_range ** val[0]
                if heat_capacity:
                    d_sum_heat_capacities -= val[0] * (val[0] - 1) * val[1] * temp_range ** (val[0] - 1)

            # Calculate the additional parts in the equation (for elements for which it exists). cfs_dict stores
            # the coefficients of gpres and gmag. For elements which don't have gpres and gmag in their equation,
            # they turn out to be 0.
            cfs_dict = self.get_cfs(data, i)
            if gibbs:
                gpres = self.get_gpres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'], cfs_dict['a3'],
                                       temp_range, p, cfs_dict['K0'], cfs_dict['K1'], cfs_dict['K2'], cfs_dict['n'])
                gmag = self.get_gmag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                gibbs_energies_total = np.concatenate((gibbs_energies_total, c * temp_range * np.log(temp_range) +
                                                       d_sum_gibbs + gpres + gmag), axis=None)
            if entropy:
                spres = self.get_spres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'], cfs_dict['a3'],
                                       temp_range, p)
                smag = self.get_smag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                entropies_total = np.concatenate((entropies_total, -c - c * np.log(temp_range) + d_sum_entropies +
                                                  spres + smag), axis=None)

            if enthalpy:
                hpres = self.get_hpres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'], cfs_dict['a3'],
                                       temp_range, p)
                hmag = self.get_hmag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                enthalpies_total = np.concatenate((enthalpies_total, -c * temp_range + d_sum_enthalpies + hpres + hmag))

            if heat_capacity:
                cpres = self.get_cpres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'], cfs_dict['a3'],
                                       temp_range, p)
                cmag = self.get_cmag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                heat_capacities_total = np.concatenate((heat_capacities_total, -c + d_sum_heat_capacities + cpres + cmag))

        # Store the obtained data to a DataFrame
        result_data = {'Temperature': temp_ranges_total}
        if gibbs:
            result_data['G(J)_' + phase] = gibbs_energies_total
        if entropy:
            result_data['S(J/K)_' + phase] = entropies_total
        if enthalpy:
            result_data['H(J)_' + phase] = enthalpies_total
        if heat_capacity:
            result_data['Cp(J/K)_' + phase] = heat_capacities_total

        gibbs_data_frame = pd.DataFrame(result_data)

        return gibbs_data_frame

    @staticmethod
    def get_temp_ranges(data, temp):
        """
        For a given data set, return the temperature ranges in which the equations are defined.
        :param data: DataFrame containing the coefficients
        :param temp: Total temperature range
        :return:
        """

        temp_ranges = []
        # First remove all temperature values below the lowest starting temperatures
        temp = temp[temp >= data['Start temperature'].min()]
        # Then remove all temperature values which are above the highest end temperatures
        temp = temp[temp <= data['End temperature'].max()]
        # Split the temperature into the ranges
        for interval_start, interval_end in zip(data['Start temperature'], data['End temperature']):
            temp_ranges.append(temp[(temp > interval_start) & (temp <= interval_end)])

        return temp_ranges

    @staticmethod
    def get_gpres(A=0, a0=0, a1=0, a2=0, a3=0, T=0, p=0, k0=0, k1=0, k2=0, n=0):
        """
        Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants

        :param A:
        :param a0:
        :param a1:
        :param a2:
        :param a3:
        :param T: temperature
        :param p: pressure
        :param k0:
        :param k1:
        :param k2:
        :param n:
        :return:
        """

        num = A * p * (1 + a0 * T + a1 * T ** 2/2 + a2 * T ** 3/3 + a3/T)

        if k0 == 0 and k1 == 0 and k2 == 0:
            return num
        elif n != 0:
            k = k0 + k1 * T + k2 * T ** 2

            num = A * np.exp(num/(A * p) - 1)

            return num/(k * (n - 1)) * ((1 + n * p * k) ** (1 - 1/n) - 1)
        else:
            raise ValueError('From the given coefficients, Gpres could not be calculated.')

    @staticmethod
    def get_gmag(T=0, p=0, Tcrit=0, B0=0):
        """
        Calculates the magnetic contribution to the Gibbs energy. The parameters are material dependent constants.
        :param T: temperature
        :param p: fraction of the magnetic enthalpy absorbed above the critical temperature
        :param Tcrit: critical temperature
        :param B0: average magnetic moment per atom
        :return:
        """

        if Tcrit == 0 or p == 0:
            return 0

        R = 8.314  # J/(mol * K)

        g_tau = np.zeros_like(T)
        D = 518 / 1125 + 11692 / 15975 * (1 / p - 1)

        tau = T/Tcrit

        g_tau[T > Tcrit] = -(tau[T > Tcrit] ** -5/10 + tau[T > Tcrit] ** -15/315 + tau[T > Tcrit] ** -25/1500) / D
        g_tau[T <= Tcrit] = 1 - (79 / (140 * tau[T <= Tcrit] * p) + 474 / 497 * (1 / p - 1) * (
                            tau[T <= Tcrit] ** 3 / 6 + tau[T <= Tcrit] ** 9 / 135 + tau[T <= Tcrit] ** 15 / 600)) / D

        return R * T * np.log(B0 + 1) * g_tau

    @staticmethod
    def get_spres(A=0, a0=0, a1=0, a2=0, a3=0, T=0, p=0):
        """
        Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants
        :param A:
        :param a0:
        :param a1:
        :param a2:
        :param a3:
        :return:
        """

        return -A * p * (a0 + a1 * T + a2 * T ** 2 - a3 * T ** -2)

    @staticmethod
    def get_smag(T=0, p=0, Tcrit=0, B0=0):
        """
        Calculates the magnetic contribution to the entropy. The parameters are material dependent constants.
        :param T: temperature
        :param p: fraction of the magnetic enthalpy absorbed above the critical temperature
        :param Tcrit: critical temperature
        :param B0: average magnetic moment per atom
        :return:
        """

        if Tcrit == 0 or p == 0:
            return 0

        R = 8.314  # J/(mol * K)

        f_tau = np.zeros_like(T)
        D = 518 / 1125 + 11692 / 15975 * (1 / p - 1)

        tau = T / Tcrit

        f_tau[T > Tcrit] = -(2 * tau[T > Tcrit] ** -5 / 5 + 2 * tau[T > Tcrit] ** -15 / 45 + 2 * tau[T > Tcrit] ** -25 / 125) / D
        f_tau[T <= Tcrit] = 1 - (474 / 497 * (1 / p - 1) * (2 * tau[T <= Tcrit] ** 3 / 3 + 2 * tau[T <= Tcrit] ** 9 / 27
                                                            + 2 * tau[T <= Tcrit] ** 15 / 75)) / D

        return - R * np.log(B0 + 1) * f_tau

    @staticmethod
    def get_hpres(A=0, a0=0, a1=0, a2=0, a3=0, T=0, p=0):
        """
        Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants
        :param A:
        :param a0:
        :param a1:
        :param a2:
        :param a3:
        :param T: temperature
        :param p: pressure
        :return:
        """

        return A * p * (1 - a1 * T ** 2/2 - 2 * a2 * T ** 3/3 + 2 * a3 * T ** -1)

    @staticmethod
    def get_hmag(T=0, p=0, Tcrit=0, B0=0):
        """
        Calculates the magnetic contribution to the entropy. The parameters are material dependent constants.
        :param T: temperature
        :param p: fraction of the magnetic enthalpy absorbed above the critical temperature
        :param Tcrit: critical temperature
        :param B0: average magnetic moment per atom
        :return:
        """

        if Tcrit == 0 or p == 0:
            return 0

        R = 8.314  # J/(mol * K)

        h_tau = np.zeros_like(T)
        D = 518 / 1125 + 11692 / 15975 * (1 / p - 1)

        tau = T / Tcrit

        h_tau[T > Tcrit] = -(tau[T > Tcrit] ** -5 / 2 + tau[T > Tcrit] ** -15 / 21 + tau[T > Tcrit] ** -25 / 60) / D
        h_tau[T <= Tcrit] = (-79/(tau[T <= Tcrit] * 140 * p) + 474 / 497 * (1 / p - 1) * (tau[T <= Tcrit] ** 3 / 2 +
                                tau[T <= Tcrit] ** 9 / 15 + tau[T <= Tcrit] ** 15 / 40)) / D

        return - R * T * np.log(B0 + 1) * h_tau

    @staticmethod
    def get_cpres(A=0, a0=0, a1=0, a2=0, a3=0, T=0, p=0):
        """
        Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants
        :param A:
        :param a0:
        :param a1:
        :param a2:
        :param a3:
        :param T: temperature
        :param p: pressure
        :return:
        """

        return -A * p * (a1 * T + 2 * a2 * T ** 2 + 2 * a3 * T ** -2)

    @staticmethod
    def get_cmag(T=0, p=0, Tcrit=0, B0=0):
        """
        Calculates the magnetic contribution to the entropy. The parameters are material dependent constants.
        :param T: temperature
        :param p: fraction of the magnetic enthalpy absorbed above the critical temperature
        :param Tcrit: critical temperature
        :param B0: average magnetic moment per atom
        :return:
        """

        if Tcrit == 0 or p == 0:
            return 0

        R = 8.314  # J/(mol * K)

        c_tau = np.zeros_like(T)
        D = 518 / 1125 + 11692 / 15975 * (1 / p - 1)

        tau = T / Tcrit

        c_tau[T > Tcrit] = (2 * tau[T > Tcrit] ** -5 + 2 * tau[T > Tcrit] ** -15 / 3 + 2 * tau[T > Tcrit] ** -25 / 5) / D
        c_tau[T <= Tcrit] = (474 / 497 * (1 / p - 1) * (2 * tau[T <= Tcrit] ** 3 + 2 * tau[T <= Tcrit] ** 9 / 3 +
                                                        2 * tau[T <= Tcrit] ** 15 / 5)) / D

        return R * np.log(B0 + 1) * c_tau

    def get_cfs(self, data, i):
        cfs = ['A', 'a0', 'a1', 'a2', 'a3', 'K0', 'K1', 'K2', 'n', 'Tcrit', 'B0']
        cfs_dict = {}
        data_cols = data.columns.values

        for cf in cfs:
            if cf in data_cols:
                cfs_dict[cf] = float(data[cf][i])
            else:
                cfs_dict[cf] = 0

        if self.element == 'Fe' and data['Phase'][i] == 'BCC_A2':
            cfs_dict['p'] = 0.4
        else:
            cfs_dict['p'] = 0.28

        return cfs_dict

    def get_gibbs_at_temp(self, temp):
        """
        Returns the Gibbs energy at a certain temperature. TODO: Interpolate if temp lies between two values
        :param temp: temperature value
        :return:
        """

        print(self.equation_result_data[self.equation_result_data['Temperature'] == temp])
