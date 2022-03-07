import PyPDF2
import pandas as pd
import os


class SGTEConverter(object):
    """
    Converts the SGTE Data PDF into excel tables with the corresponding coefficients


    """

    def __init__(self, file_location):
        super(SGTEConverter, self).__init__()

        self.file_location = file_location

        self.current_element = None
        self.last_phase = None
        self.data_df = None
        self.index = 0

        self.elements = ['Ag  ', 'Al  ', 'Am  ', 'As  ', 'Au  ', 'B  ', 'Ba  ', 'Be  ', 'Bi  ', 'C  ', 'Ca  ', 'Cd  ',
                         'Ce  ', 'Co  ', 'Cr  ', 'Cs  ', 'Cu  ', 'Dy  ', 'Er  ', 'Eu  ', 'Fe  ', 'Ga  ', 'Gd  ', 'Ge  ',
                         'Hf  ', 'Hg  ', 'Ho  ', 'In  ', 'Ir  ', 'K  ', 'La  ', 'Li  ', 'Lu  ', 'Mg  ', 'Mn  ', 'Mo  ',
                         'Na  ', 'Nb  ', 'Nd  ', 'Ni  ', 'Np  ', 'Os  ', 'P  ', 'Pa  ', 'Pb  ', 'Pd  ', 'Pr  ', 'Pt  ',
                         'Pu  ', 'Rb  ', 'Re  ', 'Rh  ', 'Ru  ', 'S  ', 'Sb  ', 'Sc  ', 'Se  ', 'Si  ', 'Sm  ', 'Sn  ',
                         'Sr  ', 'Ta  ', 'Tb  ', 'Tc  ', 'Te  ', 'Th  ', 'Ti  ', 'Tl  ', 'Tm  ', 'U  ', 'V  ', 'W  ',
                         'Y  ', 'Yb  ', 'Zn  ', 'Zr  ']
        self.phases = ['  FCC_A1  ', '\n FCC_A1 ', 'LIQUID  ', '\n LIQUID ', '  HCP_A3  ', '\n HCP_A3 ',
                       '  BCC_A2  ', '\n BCC_A2 ', '  BCT_A5  ', '\n BCT_A5 ', '  CUB_A13  ', '\n CUB_A13 ',
                       '  HCP_ZN  ', '\n HCP_ZN ', '  OMEGA  ', '\n OMEGA', '  ORTHORHOMBIC_A20  ',
                       '\n ORTHORHOMBIC_A20 ',
                       '  TETRAGONAL_U  ', '\n TETRAGONAL_U ', '  CBCC_A12  ', '\n CBCC_A12 ',
                       '  DIAMOND_A4  ', '\n DIAMOND_A4 ', '  DHCP  ', '\n DHCP ', '  RHOMBOHEDRAL_A7  ',
                       '\n RHOMBOHEDRAL_A7 ',
                       '  RED_P  ', '\n RED_P ', '  BETA_RHOMBO_B  ', '\n BETA_RHOMBO_B ', '  GRAPHITE  ',
                       '\n GRAPHITE ',
                       '  TETRAGONAL_A6  ', '\n TETRAGONAL_A6 ', '  TET_ALPHA1  ', '\n TET_ALPHA1 ', '  RHOMBO_A10  ',
                       '\n RHOMBO_A10 ', '  ORTHORHOMBIC_GA  ', '\ ORTHORHOMBIC_GA ', '  GAS (1/2N2)  ',
                       '\n GAS (1/2N2) ',
                       '  ORTHO_Ac  ', '\n ORTHO_Ac ', '  TETRAG_Ad  ', '\n TETRAG_Ad ', '  GAS (1/2O2<g>)  ',
                       '\n GAS (1/2O2<g>) ', '  WHITE_P  ', '\n WHITE_P ', '  BCT_Aa  ', '\n BCT_Aa ', '  ALPHA_PU  ',
                       '\n ALPHA_PU ', '  BETA_PU  ', '\n BETA_PU ', '  GAMMA_PU  ', '\n GAMMA_PU ',
                       '  ORTHORHOMBIC_S  ',
                       '\n ORTHORHOMBIC_S ', '  MONOCLINIC  ', '\n MONOCLINIC ', '  HEXAGONAL_A8  ', '\n HEXAGONAL_A8 ',
                       '  RHOMB_C19  ', '\n RHOMB_C19 ']

    def read_doc(self):
        """ """
        pdf_file_obj = open(self.file_location, 'rb')
        pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)

        # for p in range(12, 172):
        for p in range(170, 172):
            page = pdf_reader.getPage(p)
            page_text = page.extractText()

            for element in self.elements:
                if page_text.find(element) > -1:
                    if self.current_element is not None:
                        file_path = os.path.join(r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4 Daten\SGTE Data",
                                                 self.current_element.strip() + ".xlsx")
                        if self.data_df is not None:
                            self.data_df.to_excel(file_path, index=False)
                        else:
                            print('NO DATAFRAME')
                        self.index = 0
                        self.data_df = None

                    self.current_element = element
                    print()
                    print('----------')
                    print(self.current_element, ', ', p)
                    print('----------')
                    print()
                    self.elements.remove(element)

            slices = []
            loc_data_rel = page_text.find('Data relative to')
            if loc_data_rel == -1:
                loc_data_rel = len(page_text)

            for phase in self.phases:
                text_loc = page_text.find(phase)
                if -1 < text_loc < loc_data_rel:
                    if len(slices) == 0:
                        test_split = page_text[:text_loc].split(' ')
                        try:
                            float(test_split[0])
                            slices.append((0, self.last_phase))
                        except:
                            pass
                    slices.append((text_loc, phase))
                else:
                    phase = phase.strip()
                    stripped_text_loc = page_text.find(phase)
                    if stripped_text_loc == 0:
                        slices.append((stripped_text_loc, phase))
            slices.append((loc_data_rel, 'End'))

            if slices[0][1] == 'End':
                slices.append((0, self.last_phase))

            slices.sort(key=lambda y: y[0])
            print(slices)

            for i, slc in enumerate(slices):
                if i < len(slices) - 1:
                    split_text = page_text[slc[0]:slices[i + 1][0]]
                    split_text = split_text.split(' ')
                    print(split_text)
                    self.last_phase = slc[1]
                    self.extract_text(split_text, slc[1])

    def extract_text(self, text, phase):
        """

        Parameters
        ----------
        text :

        phase :


        Returns
        -------

        """
        sign = 1
        text_dict = {'Phase': phase.strip()}
        cfs = ['a0', 'a1', 'a2', 'a3', 'K0', 'K1', 'K2', 'TC', 'TN', 'B0']

        for i, t in enumerate(text):
            # Extract the coefficients with the correct sign
            if t == '+':
                sign = 1
            elif t == '-':
                sign = -1

            try:
                text[i] = sign * float(t)

                if text[i + 1] == '+' or text[i + 1] == '-':
                    text_dict['0'] = text[i]
            except:
                pass

            if '\n' in t:
                t = t.replace('\n', '')

            # Extract the exponents
            if 'T' in t and len(t) > 1 and t != 'ln(T)' and t != 'TN' and t != 'TC':
                exp = t.replace('T', '')
                try:
                    i_exp = int(exp)
                    text_dict[str(i_exp)] = text[i - 1]
                except:
                    pass
            elif 'T' in t and len(t) == 1 and text[i + 1] != 'ln(T)' and text[i - 1] != '<':
                text_dict['1'] = text[i - 1]
            elif t == 'ln(T)':
                text_dict['c'] = text[i - 2]
            elif t == 'A':
                text_dict['A'] = text[i + 3]
            elif t == 'n':
                if text[i + 1] == '':
                    text_dict['n'] = text[i + 3]
                else:
                    text_dict['n'] = text[i + 2]
            elif t in cfs:
                if t == 'TN' or t == 'TC':
                    t = 'Tcrit'
                text_dict[t] = text[i + 2]
            elif '(' in t:
                text_dict['Start temperature'] = t.replace('(', '')
            elif ')' in t:
                text_dict['End temperature'] = t.replace(')', '')
                if len(text_dict) > 1:
                    if self.data_df is None:
                        self.data_df = pd.DataFrame(text_dict, index=[self.index])
                    else:
                        self.data_df = self.data_df.append(pd.DataFrame(text_dict, index=[self.index]))
                    self.index += 1

                new_text_dict = {'Phase': phase.strip()}
                if 'A' in text_dict:
                    new_text_dict['A'] = text_dict['A']
                if 'n' in text_dict:
                    new_text_dict['n'] = text_dict['n']
                if 'Tcrit' in text_dict:
                    new_text_dict['Tcrit'] = text_dict['Tcrit']

                for cf in cfs:
                    if cf in text_dict:
                        new_text_dict[cf] = text_dict[cf]

                text_dict = new_text_dict

                sign = 1


sgte = SGTEConverter(r"C:\Users\danie\Documents\Montanuni\Masterarbeit\2 Literatur\Daten\SGTE Data For Pure Elements [Dinsdale, A.].pdf")
sgte.read_doc()
