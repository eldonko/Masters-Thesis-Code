import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyPDF2


class DataHandler(object):
    """
    Given data from different sources, DataConverter converts and reformats the data so
    that it can be used in training a neural network. The data can be saved as csv file or
    returned as a pandas DataFrame.

    File naming convention: Element/Compound_lower temperature_upper temperature_temperature increment
    The temperatures in the filenames are in K for all elements.

    Table 1:
    -----------------------------
    Data Source     | File format
    -----------------------------
    FactSage        | .txt
    """

    def __init__(self, source, input_file):
        """

        :param source: name string of the data source as listed in Table 1.
        :param input_file: absolute path of input file
        """
        super(DataHandler, self).__init__()

        # Specify error prompts
        self.errors = {'ERR_VIS_1': '*** ERR_VIS_1: Entered column name is not a valid column name *** ',
                       'ERR_SRC_1': '*** ERR_SRC_1: Entered source is not a valid source ***',
                       'ERR_SRC_2': '*** ERR_SRC_2: Entered file could not be opened ***',
                       'ERR_SVE_1': '*** ERR_SVE_1: Table could not be saved because input_file is None',
                       'ERR_SVE_2': '*** ERR_SVE_2: Table could not be saved because source is None'}

        self.input_file = input_file
        self.table = None
        self.source = None
        self.possible_sources = [None, 'FactSage', 'Barin', 'Dorog']
        self.set_source(source)
        self.id_figure = 1  # Used in specifying a new plot
        self.max_suffix = 1

        if self.source is not None and self.input_file is not None:
            self.load_table()

    def convert(self, set_table=True):
        """
        Loads and converts the table

        :param set_table: If set table is True, the converted table will be saved to self.table.
        If set_table is False, the converted table is returned as pandas DataFrame. This is because
        converting table might be necessary without wanting to change self.table
        :return:
        """

        def text_from_pdf(pdf_file, page_number):
            # creating a pdf file object
            pdf_file_obj = open(pdf_file, 'rb')

            # creating a pdf reader object
            pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)

            # creating a page object
            page_obj = pdf_reader.getPage(page_number)

            return page_obj.extractText()

        # Treat different sources differently
        if self.source == 'FactSage':
            # Extract the data and temporarily save every column in the txt file into a list
            first = True
            end_of_block = False
            header = 5
            table_dict = dict()
            save_data = []
            column_names = []

            phase_translation = {'Fe': {'S1': 'bcc', 'S2': 'fcc', 'L1': 'liquid', 'G1': 'gas'}}

            element = self.get_element_from_file()

            def table_to_dict():
                """
                Takes the column names and the extracted data and makes a dict of it (easy to convert
                into pd.DataFrame)
                :return:
                """
                for i, c in enumerate(column_names):
                    table_dict[c] = save_data[i]

            with open(self.input_file, 'r') as f:
                for i, line in enumerate(f):
                    #print(i, 'O', line)
                    if i < header:
                        continue

                    # FactSage tables are separated by a row of '_'. If such row occurs, save
                    # the data obtained priorly and reinitialize save_data.
                    if '_' in line:
                        # If the '_' row occurs the first time, there is nothing yet to save
                        if not first and not end_of_block:
                            table_to_dict()
                        elif first:
                            first = False

                        # The row of '_' marks the end and the beginning of a block. In between
                        # are the column names. To extract those, end_of_block marks which row
                        # of '_' was found (first or second).
                        end_of_block = not end_of_block
                    # If end_of_block is True, the line is in between two lines of '_'. Then,
                    # the column names can be extracted.
                    elif end_of_block:
                        # Get the column names
                        column_names = list(filter(lambda v: v != '', line.strip().split(' ')))
                        # Get the phase (first entry in column_names) and remove it
                        phase = column_names[0]
                        column_names.remove(phase)
                        # If there is a different name for the phase, translate it
                        if element in phase_translation:
                            phase = phase_translation[element][phase]
                        # Append _phase to every column name to make it unique (there might be other phases)
                        column_names = [c + '_' + phase for c in column_names]

                        # Reset save_data
                        save_data = []
                        for _ in range(len(column_names)):
                            save_data.append([])
                    # If an empty line is found, EOF is reached
                    elif line.isspace():
                        # If EOF is reached, table_dict is converted to a pandas DataFrame
                        table_to_dict()
                        table = pd.DataFrame(table_dict)
                        if set_table:
                            self.table = table
                        else:
                            return table
                    # Extract the numeric values
                    else:
                        # Get the values
                        values = list(filter(lambda v: v != '' and not v.isalpha(), line.strip().split(' ')))
                        for i_data in range(len(save_data)):
                            save_data[i_data].append(float(values[i_data]))

    def get_element_from_file(self):
        """
        Extracts the element from the filename. The element is always specified as the string after the first '_'
        in the filename
        :return: String of element abbreviation
        """

        return self.input_file.split('_')[1]

    def save_table(self, output_filename=None):
        """

        :param output_filename: Specifies where to save the table at. Is None by default, which indicates
        that the file should be saved in the standard location defined in code
        :return:
        """

        if output_filename is None:
            # Check if there is an input file specified and ask for one if not
            if self.input_file is None:
                print(self.errors['ERR_SVE_1'])
                self.input_file = input('Please enter absolute input file path: ')
            # Check if there is a source specified and ask for one if not
            if self.source is None:
                print(self.errors['ERR_SVE_2'])
                self.set_source(-1)

            directory = r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4 Daten\Preprocessed" + "\\" \
                        + self.get_element_from_file()
            filename = os.path.basename(self.input_file).split('.')[0] + '.xlsx'
            self.table.to_excel(os.path.join(directory, filename), index=False)
        else:
            self.table.to_excel(output_filename, index=False)

    def get_x_y_data(self, x_col, y_col):
        """

        :param x_col: column name of x values
        :param y_col: column name of y values
        :return: x, y data as numpy arrays
        """

        self.load_table()

        # Extract the x and y data
        x = np.array(self.table[x_col])
        y = np.array(self.table[y_col])

        return x, y

    def get_column_names(self):
        self.load_table()

        cols = self.table.columns
        print(list(cols))

        return cols

    def set_input_file(self, new_input_file):
        # Check if file exists
        if os.path.isfile(new_input_file):
            self.input_file = new_input_file
        else:
            print(self.errors['ERR_SRC_2'])
            while not os.path.isfile(new_input_file):
                new_input_file = input('Enter file path: ')
            self.input_file = new_input_file

    def set_source(self, new_source):
        # Check if source is valid
        while new_source not in self.possible_sources:
            print(self.errors['ERR_SRC_1'])
            print('Source entered: ', new_source, ' Please select one of the following sources:')
            print(self.possible_sources)
            new_source = input('Enter source: ')
        self.source = new_source

    def show_table(self, n_cols=5):
        self.load_table()

        print(self.table.head(n_cols))

    def load_table(self):
        # If no table was opened yet, open the one specified
        if self.table is None:
            # If the input file is a .txt file from FactSage, it needs to be converted first
            filename, file_extension = os.path.splitext(self.input_file)
            if self.source == 'FactSage' and file_extension.lower() == '.txt':
                self.convert()
            elif file_extension.lower() == '.csv':
                self.table = pd.read_csv(self.input_file)
            elif file_extension.lower() == '.xlsx':
                self.table = pd.read_excel(self.input_file)

    def change_table(self, source, input_file):
        self.set_source(source)
        self.set_input_file(input_file)
        self.load_table()

    def plot_data(self, x_col, y_col, title=None, axis_labels=None):
        """
        Plots the data from the table
        :param x_col: Column name for the x-values
        :param y_col: Column name for the y-values
        :param title: Plot title
        :param axis_labels: Axis labels in the format (x_label, y_label)
        :return:
        """

        self.load_table()

        plt.figure(self.id_figure)
        self.id_figure += 1

        def check_column_name(col_name):
            new_col = col_name
            while col_name not in self.table.columns:
                print(self.errors['ERR_VIS_1'])
                print('Please enter valid column name! Valid columns:')
                self.get_column_names()
                new_col = input('Enter column name: ')

            return new_col

        # Check x_col
        x_col = check_column_name(x_col)

        # Check y_col(s) and plot the data
        if type(y_col) == list:
            for i, col in enumerate(y_col):
                y_col[i] = check_column_name(col)
                x, y = self.get_x_y_data(x_col, y_col[i])
                plt.scatter(x, y, s=1)
                plt.legend(y_col)
        else:
            y_col = check_column_name(y_col)
            x, y = self.get_x_y_data(x_col, y_col)
            plt.scatter(x, y, s=1)

        # Beautify the plot
        plt.grid(b=True)
        if title is not None:
            plt.title(title)
        else:
            plt.title(None)

        if axis_labels is not None:
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
        else:
            plt.xlabel(None)
            plt.ylabel(None)

    def plot_gibbs(self, temp_col_name=None):
        """
        Makes a full plot of all the Gibbs energies for all phases over the temperature
        :param temp_col_name: Specifies a column which includes temperature values
        :return:
        """

        self.plot_variable_over_temp('G(J)', plot_title='Gibbs energy vs temperature', temp_col_name=None)

    def plot_enthalpy(self, temp_col_name=None):
        """
        Makes a full plot of all the Enthalpies for all phases over the temperature
        :param temp_col_name: Specifies a column which includes temperature values
        :return:
        """

        self.plot_variable_over_temp('H(J)', plot_title='Enthalpy vs temperature', temp_col_name=None)

    def plot_heat_capacity(self, temp_col_name=None):
        """
        Makes a full plot of all the heat capacities for all phases over the temperature
        :param temp_col_name: Specifies a column which includes temperature values
        :return:
        """

        self.plot_variable_over_temp('Cp(J/K)', plot_title='Heat capacity vs temperature', temp_col_name=None)

    def plot_entropy(self, temp_col_name=None):
        """
        Makes a full plot of the Entropy for all phases over the temperature
        :param temp_col_name: Specifies a column which includes temperature values
        :return:
        """

        self.plot_variable_over_temp('S(J/K)', plot_title='Entropy vs temperature', temp_col_name=None)

    def plot_variable_over_temp(self, y_col_name: str, plot_title=None, temp_col_name=None):
        """
        Plots all columns where the name contains y_col_name (a thermodynamic variable like the
        Gibbs energy) over the temperature

        :param y_col_name: Specifies which columns shall be plotted. For example for plotting
        all Gibbs energy columns, col_name could be 'G(J)'
        :param plot_title: Specifies the title of the plot
        :param temp_col_name: Specifies a column which includes temperature values
        :return:
        """

        self.load_table()

        columns = self.table.columns

        # Find temperature column and all columns with Gibbs energies of different phases
        var_cols = []
        for col in columns:
            if 'T(K)' in col and temp_col_name is None:
                temp_col_name = col
            elif y_col_name in col:
                var_cols.append(col)

        # Don't plot, if no data is available
        if len(var_cols) == 0:
            print('No data available for ', y_col_name)
        else:
            self.plot_data(temp_col_name, var_cols, title=plot_title, axis_labels=('T(K)', y_col_name))

    def merge(self, other, keep_data_handler=True, keep_tables_unique=False):
        """

        :param keep_data_handler: If keep is True, the DataHandler from which merge was called will now
        be the merged DataHandler (it contains the merged DataFrame)
        :param other: DataHandler to merge with
        :param keep_tables_unique: If keep_tables_unique is False, columns with the same name but from different sources
        will be mixed into one column. If keep_tables_unique is True, the columns will be kept unique by adding suffixes
        to their column names (ascending index saved in self.max_suffix)
        :return: Merged DataHandler
        """

        if not keep_tables_unique:
            merged_table = self.table.append(other.table, ignore_index=True)
            merged_table = merged_table.sort_values(by=['T(K)_bcc'])
        else:
            left_table = self.table.add_suffix('_' + str(self.max_suffix))
            right_table = self.table.add_suffix('_' + str(self.max_suffix + 1))
            self.max_suffix += 2
            merged_table = left_table.append(right_table, ignore_index=True)

        if keep_data_handler:
            self.table = merged_table
            self.set_source(None)
            self.set_input_file(None)
        else:
            new_data_handler = DataHandler(None, None)
            new_data_handler.table = merged_table
            new_data_handler.max_suffix = self.max_suffix
            return new_data_handler
