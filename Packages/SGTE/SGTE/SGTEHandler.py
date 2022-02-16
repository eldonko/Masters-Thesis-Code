import os
import pkg_resources

import numpy as np
import pandas as pd


class SGTEHandler(object):
    """
    Converts the sgte data coefficients stored in data into (temperature, value) pairs.
    """
    def __init__(self, element=None):
        super(SGTEHandler, self).__init__()
        self.data = None
        self.measurements = None
        self.colors = ['royalblue', 'seagreen', 'slategrey', 'red', 'pink', 'orange', 'mintcream', 'lime', 'yellow']
        self.element = None
        self.elements = ['Ag', 'Al', 'Am', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'C', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cs',
                         'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'Hg', 'Ho', 'In', 'Ir', 'K', 'La', 'Li',
                         'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
                         'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta',
                         'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

        self.errors = {'INP_1': '*** INP_1 *** Invalid input, please try again!'}

        self.equation_result_data = None
        self.selected_phases = []

        self.set_element(element)

    def set_element(self, element=None):
        """
        Sets the element for which the sgte equations should be evaluated

        Parameters
        ----------
        element : str
            Element for which the sgte equations should be evaluated. If None, user will be asked for input
            (Default value = None)

        Returns
        -------

        """

        # If element input is provided, set the element as class attribute
        if element is not None and element in self.elements:
            self.element = element

        # If no or malicious user input is provided, ask the user (again) for element input
        while self.element not in self.elements:
            self.element = input('Please enter element name: ').capitalize()
            print(self.element + ' successfully selected!\n')

        # Load the data
        inp_file = os.path.join('data', self.element + '.xlsx')
        self.load_data(inp_file)

    def set_phases(self, phases=None):
        """
        Sets the phases for which the sgte equations should be evaluated

        Parameters
        ----------
        phases : list
             Phases for which the sgte equations should be evaluated. If None, user will be asked for input.
             (Default value = None)

        Returns
        -------

        """
        # Get all the unique phases in self.data
        possible_phases = self.data['Phase'].unique()

        # If phases is not None, set the phases as passed. If phases is ['all'], use all possible phases
        if phases is not None:
            # TODO: Add error handling
            if phases == ['all']:
                self.selected_phases = possible_phases
            else:
                self.selected_phases = phases
            return

        # Make phases reusable of sgte is used multiple times
        if len(self.selected_phases) > 0:
            print('To reuse phases from before, enter y. To reset phases, enter n.')
            keep = ''
            while keep not in ['y', 'n']:
                keep = input('[y/n]: ').strip()
                # If input is y, return as nothing has to be changed. If input is n, proceed with phase selection
                if keep == 'y':
                    return

        # Prompt for user
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

    def solve_equations(self, phase, p, temp, gibbs, entropy, enthalpy, heat_capacity):
        """Evaluates the equations specified by Dinsdale (sgte). For further reference see "sgte Data for
        Pure Elements [Dinsdale]"

        Parameters
        ----------
        temp : np.ndarray
            temperature
        phase : str
            The phase in which the Gibbs energy is to be evaluated.
        p : float
            pressure
        gibbs : bool
            if True, Gibbs energy is evaluated
        entropy : bool
            if True, entropy is evaluated
        enthalpy : bool
            if True, enthalpy is evaluated
        heat_capacity : bool
            if True, heat capacity is evaluated

        Returns
        -------
        pd.DataFrame
            equation results

        """

        # Get the sgte coefficients from self.data
        data = self.data[self.data['Phase'] == phase].sort_values(by='Start temperature',
                                                                  ascending=True).reset_index(
            drop=True)

        # Allow for temperature ranges which go over the intervals specified in sgte data. For this, it is necessary to
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
                if heat_capacity:
                    d_sum_heat_capacities -= val[0] * (val[0] - 1) * val[1] * temp_range ** (val[0] - 1)

            # Calculate the additional parts in the equation (for elements for which it exists). cfs_dict stores
            # the coefficients of gpres and gmag. For elements which don't have gpres and gmag in their equation,
            # they turn out to be 0.
            cfs_dict = self.get_cfs(data, i)
            if gibbs:
                gpres = self.get_gpres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'],
                                       cfs_dict['a3'],
                                       temp_range, p, cfs_dict['K0'], cfs_dict['K1'], cfs_dict['K2'], cfs_dict['n'])
                gmag = self.get_gmag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                gibbs_energies_total = np.concatenate((gibbs_energies_total, c * temp_range * np.log(temp_range) +
                                                       d_sum_gibbs + gpres + gmag), axis=None)
            if entropy:
                spres = self.get_spres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'],
                                       cfs_dict['a3'],
                                       temp_range, p)
                smag = self.get_smag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                entropies_total = np.concatenate((entropies_total, -c - c * np.log(temp_range) + d_sum_entropies +
                                                  spres + smag), axis=None)

            if enthalpy:
                hpres = self.get_hpres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'],
                                       cfs_dict['a3'],
                                       temp_range, p)
                hmag = self.get_hmag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                enthalpies_total = np.concatenate(
                    (enthalpies_total, -c * temp_range + d_sum_enthalpies + hpres + hmag))

            if heat_capacity:
                cpres = self.get_cpres(cfs_dict['A'], cfs_dict['a0'], cfs_dict['a1'], cfs_dict['a2'],
                                       cfs_dict['a3'],
                                       temp_range, p)
                cmag = self.get_cmag(temp_range, cfs_dict['p'], cfs_dict['Tcrit'], cfs_dict['B0'])
                heat_capacities_total = np.concatenate(
                    (heat_capacities_total, -c + d_sum_heat_capacities + cpres + cmag))

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

    def evaluate_equations(self, start_temp, end_temp, p, gibbs=True, entropy=False, enthalpy=False, heat_capacity=False
                           , plot=True, phases=None, step=1):
        """Evaluates the Gibbs energy for a temperature range and selected phase at a certain pressure.

        Parameters
        ----------
        start_temp : int
            lower bound of the temperature interval
        end_temp : int
            upper bound of the temperature interval
        p : float
            pressure
        gibbs : bool
            if True, Gibbs energy is evaluated (Default value = True)
        entropy : bool
            if True, entropy is evaluated (Default value = False)
        enthalpy : bool
             if True, enthalpy is evaluated (Default value = False)
        heat_capacity : bool
            if True, heat capacity is evaluated (Default value = False)
        plot : bool
            if True, results are plotted (Default value = True)
        phases : list
            phases to be evaluated as list of strings or None (Default value = None)
        step : float
            step size of temperature range. Determines the amount of data created. The smaller the step size, the more
            data is created. (Default value = 1)

        Returns
        -------

        """

        # Set the phases
        self.set_phases(phases)

        # Create the temperature range
        temp_range = np.arange(start_temp, end_temp, step, dtype=np.float32)
        ax_gibbs = None
        ax_entropy = None
        ax_enthalpy = None
        ax_heat_capacity = None

        # Loop through the selected phases
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

            # Save the obtained data to the sgte
            if self.equation_result_data is None:
                self.equation_result_data = equation_result
            else:
                self.equation_result_data = self.equation_result_data.merge(equation_result, on='Temperature')

        # Plot the results
        if plot:
            if gibbs:
                ax_gibbs.legend()
            if entropy:
                ax_entropy.legend()
            if enthalpy:
                ax_enthalpy.legend()
            if heat_capacity:
                ax_heat_capacity.legend()

    def get_stable_properties(self, start_temp, end_temp, p=1e5, measurement='G'):
        """Solves the sgte equations for the given element and all phases and returns only the properties in the stable
        phases. This means, the Gibbs energy needs to be evaluated no matter which measurement is conducted, because
        the stable with the minimum Gibbs energy at a certain temperature is the stable phase.

        Parameters
        ----------
        start_temp : int
            lower value of the temperature range
        end_temp : int
            upper value of the temperature range
        p : float
            pressure (Default value = 1e5)
        measurement : str
            Measurement type/property for which the evaluation should be made. Must be one of 'G', 'S',
            'H' or 'C' (Default value = 'G')

        Returns
        -------

        """

        # Input checking
        assert measurement in ['G', 'S', 'H', 'C']

        # Based on measurement decide which properties to evaluate. Only one of the three can be True
        entropy = True if measurement == 'S' else False
        enthalpy = True if measurement == 'H' else False
        heat_cap = True if measurement == 'C' else False

        # Evaluate the equations for the Gibbs energy and if selected another property
        self.evaluate_equations(start_temp, end_temp, p, gibbs=True, entropy=entropy, enthalpy=enthalpy,
                                heat_capacity=heat_cap, plot=False, phases=['all'])

        # Get the results
        data = self.equation_result_data

        # Extract the properties for the stable phase. If measurement is not the Gibbs energy, first the indices of
        # the minimum Gibbs energy have to be extracted so that afterwards the values at those indices can be chosen
        # from the property
        step = 0
        if measurement != 'G':
            step = 1

        # Get the Gibbs energies from the data
        gibbs_indices = list(range(1, len(data.columns), 1 + step))
        gibbs_energies = data.iloc[:, gibbs_indices]

        # Select the indices where the Gibbs energies are minimal and add 1 so that it matches with the index of
        # the same phase of the property which is to be selected
        indices = np.argmin(gibbs_energies.values, axis=1) * (1 + step) + 1 + step

        # Get the measurements from the properties where the respective phase is stable
        self.measurements = pd.DataFrame()
        self.measurements['Temperature'] = data['Temperature']
        self.measurements['Measurements'] = data.values[np.arange(len(data)), indices]

    def plot_data(self, equation_result, ax, prefix, phase, i):
        """Plots the data of the equation results

        Parameters
        ----------
        equation_result : pd.DataFrame
            results of the sgte equation evaluation
        ax : matplotlib.pyplot ax
            the axis to plot to
        prefix : str
            the prefix of the column name
        phase : str
            the evaluated phase
        i : int
            phase index

        Returns
        -------

        """

        if ax is None:
            ax = equation_result.plot(x='Temperature', y=prefix + phase, kind='scatter', grid=True, s=0.5,
                                      color=self.colors[i], label=phase, ylabel=prefix.replace('_', ''))
        else:
            equation_result.plot(x='Temperature', y=prefix + phase, kind='scatter', grid=True, s=0.5, ax=ax,
                                 color=self.colors[i], label=phase, ylabel=prefix.replace('_', ''))

        return ax

    def load_data(self, input_file):
        """Loads the data from a specified excel file and fills all NaN values with 0

        Parameters
        ----------
        input_file : str
            Excel file path which contains sgte Data

        Returns
        -------

        """
        #self.data = pd.read_excel(input_file)
        stream = pkg_resources.resource_stream(__name__, input_file)
        self.data = pd.read_excel(stream)
        self.data = self.data.fillna(0)

        # Convert the column names which are integers from str to int
        for col in self.data.columns.values:
            try:
                i_col = int(col)
                self.data = self.data.rename(columns={col: i_col})
            except:
                pass

    @staticmethod
    def get_temp_ranges(data, temp):
        """For a given data set, return the temperature ranges in which the equations are defined.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the coefficients
        temp : np.ndarray
            Total temperature range

        Returns
        -------
        list
            temperature range
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
        """Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants

        Parameters
        ----------
        A : float
            param a0: (Default value = 0)
        a1 : float
            param a2: (Default value = 0)
        a3 : float
            param T: temperature (Default value = 0)
        p : float
            pressure (Default value = 0)
        k0 : float
            param k1: (Default value = 0)
        k2 : float
            param n: (Default value = 0)
        a0 : float
             (Default value = 0)
        a2 : float
             (Default value = 0)
        T : np.ndarray
             (Default value = 0)
        k1 : float
             (Default value = 0)
        n : float
             (Default value = 0)

        Returns
        -------
        np.ndarray
            gpres term of sgte equation

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
        """Calculates the magnetic contribution to the Gibbs energy. The parameters are material dependent constants.

        Parameters
        ----------
        T : np.ndarray
            temperature (Default value = 0)
        p : float
            fraction of the magnetic enthalpy absorbed above the critical temperature (Default value = 0)
        Tcrit : float
            critical temperature (Default value = 0)
        B0 : float
            average magnetic moment per atom (Default value = 0)

        Returns
        -------
        np.ndarray
            gmag term of sgte equations

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
        """Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants

        Parameters
        ----------
        A : float
            param a0: (Default value = 0)
        a1 : float
            param a2: (Default value = 0)
        a3 : float
            return: (Default value = 0)
        a0 : float
             (Default value = 0)
        a2 : float
             (Default value = 0)
        T : np.ndarray
             (Default value = 0)
        p : float
             (Default value = 0)

        Returns
        -------
        np.ndarray
            spres term of sgte equations

        """

        return -A * p * (a0 + a1 * T + a2 * T ** 2 - a3 * T ** -2)

    @staticmethod
    def get_smag(T=0, p=0, Tcrit=0, B0=0):
        """Calculates the magnetic contribution to the entropy. The parameters are material dependent constants.

        Parameters
        ----------
        T : np.ndarray
            temperature (Default value = 0)
        p : float
            fraction of the magnetic enthalpy absorbed above the critical temperature (Default value = 0)
        Tcrit : float
            critical temperature (Default value = 0)
        B0 : float
            average magnetic moment per atom (Default value = 0)

        Returns
        -------
        np.ndarray
            smag term of sgte equations

        """

        if Tcrit == 0 or p == 0:
            return 0

        R = 8.314  # J/(mol * K)

        f_tau = np.zeros_like(T)
        D = 518 / 1125 + 11692 / 15975 * (1 / p - 1)

        tau = T / Tcrit

        f_tau[T > Tcrit] = (2 * tau[T > Tcrit] ** -5 / 5 + 2 * tau[T > Tcrit] ** -15 / 45 + 2 * tau[T > Tcrit] ** -25 / 125) / D
        f_tau[T <= Tcrit] = 1 - (474 / 497 * (1 / p - 1) * (2 * tau[T <= Tcrit] ** 3 / 3 + 2 * tau[T <= Tcrit] ** 9 / 27
                                                            + 2 * tau[T <= Tcrit] ** 15 / 75)) / D

        return - R * np.log(B0 + 1) * f_tau

    @staticmethod
    def get_hpres(A=0, a0=0, a1=0, a2=0, a3=0, T=0, p=0):
        """Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants

        Parameters
        ----------
        A : float
            param a0: (Default value = 0)
        a1 : float
            param a2: (Default value = 0)
        a3 : float
            param T: temperature (Default value = 0)
        p : float
            pressure (Default value = 0)
        a0 : float
             (Default value = 0)
        a2 : float
             (Default value = 0)
        T : np.ndarray
             (Default value = 0)

        Returns
        -------
        np.ndarray
            hpres term of sgte equations

        """

        return A * p * (1 - a1 * T ** 2/2 - 2 * a2 * T ** 3/3 + 2 * a3 * T ** -1)

    @staticmethod
    def get_hmag(T=0, p=0, Tcrit=0, B0=0):
        """Calculates the magnetic contribution to the entropy. The parameters are material dependent constants.

        Parameters
        ----------
        T : np.ndarray
            temperature (Default value = 0)
        p : float
            fraction of the magnetic enthalpy absorbed above the critical temperature (Default value = 0)
        Tcrit : float
            critical temperature (Default value = 0)
        B0 : float
            average magnetic moment per atom (Default value = 0)

        Returns
        -------
        np.ndarray
            hmag term of sgte equations

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
        """Calculates the pressure dependent contribution to the Gibbs energy. The parameters are material dependent
        constants

        Parameters
        ----------
        A : float
            param a0: (Default value = 0)
        a1 : float
            param a2: (Default value = 0)
        a3 : float
            param T: temperature (Default value = 0)
        p : float
            pressure (Default value = 0)
        a0 : float
             (Default value = 0)
        a2 : float
             (Default value = 0)
        T : np.ndarray
             (Default value = 0)

        Returns
        -------
        np.ndarray
            cpres term of sgte equations

        """

        return -A * p * (a1 * T + 2 * a2 * T ** 2 + 2 * a3 * T ** -2)

    @staticmethod
    def get_cmag(T=0, p=0, Tcrit=0, B0=0):
        """Calculates the magnetic contribution to the entropy. The parameters are material dependent constants.

        Parameters
        ----------
        T : np.ndarray
            temperature (Default value = 0)
        p : float
            fraction of the magnetic enthalpy absorbed above the critical temperature (Default value = 0)
        Tcrit : float
            critical temperature (Default value = 0)
        B0 : float
            average magnetic moment per atom (Default value = 0)

        Returns
        -------
        np.ndarray
            cmag term of sgte equations

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
        """
        Loads the coefficients for the terms (gpres, gmag, ..., cmag)

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the coefficients for an element and the respective phases in a temperature range
            
        i : int
            phase index in data

        Returns
        -------
        pd.DataFrame
            DataFrame containing the coefficients for the terms (gpres, gmag, ..., cmag)
        """
        cfs = ['A', 'a0', 'a1', 'a2', 'a3', 'K0', 'K1', 'K2', 'n', 'Tcrit', 'B0']
        cfs_dict = {}
        data_cols = data.columns.values

        for cf in cfs:
            if cf in data_cols:
                cfs_dict[cf] = float(data[cf][i])
            else:
                cfs_dict[cf] = 0

        # Treat the special case of BCC_A2 phase of Fe according to Dinsdale
        if self.element == 'Fe' and data['Phase'][i] == 'BCC_A2':
            cfs_dict['p'] = 0.4
        else:
            cfs_dict['p'] = 0.28

        return cfs_dict

    def get_gibbs_at_temp(self, temp):
        """Returns the Gibbs energy at a certain temperature. TODO: Interpolate if temp lies between two values

        Parameters
        ----------
        temp : float
            temperature value

        Returns
        -------

        """

        print(self.equation_result_data[self.equation_result_data['Temperature'] == temp])
