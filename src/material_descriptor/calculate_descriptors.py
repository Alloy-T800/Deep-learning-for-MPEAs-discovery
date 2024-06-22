import math

# Create a dictionary to store the attributes of an element
element_properties = {
    'Fe': {'atomic_weight': 55.845, 'density': 7.87, 'heat_capacity': 25.1, 'thermal_conductivity': 80.4, 'atomic_radius': 126, 'electronegativity_P': 1.83, 'electronegativity_A': 1.61, 'VEC': 8, 'melting_point': 1811, 'shear_modulus': 82},
    'Co': {'atomic_weight': 58.933, 'density': 8.86, 'heat_capacity': 24.81, 'thermal_conductivity': 100, 'atomic_radius': 125, 'electronegativity_P': 1.88, 'electronegativity_A': 1.7, 'VEC': 9, 'melting_point': 1768, 'shear_modulus': 75},
    'Ni': {'atomic_weight': 58.6934, 'density': 8.908, 'heat_capacity': 26.07, 'thermal_conductivity': 90.9, 'atomic_radius': 124, 'electronegativity_P': 1.91, 'electronegativity_A': 1.75, 'VEC': 10, 'melting_point': 1728, 'shear_modulus': 76},
    'Cr': {'atomic_weight': 51.9961, 'density': 7.19, 'heat_capacity': 23.35, 'thermal_conductivity': 93.9, 'atomic_radius': 128, 'electronegativity_P': 1.66, 'electronegativity_A': 1.56, 'VEC': 6, 'melting_point': 1907, 'shear_modulus': 115},
    'Mn': {'atomic_weight': 54.938045, 'density': 7.21, 'heat_capacity': 26.32, 'thermal_conductivity': 7.81, 'atomic_radius': 127, 'electronegativity_P': 1.55, 'electronegativity_A': 1.61, 'VEC': 7, 'melting_point': 1519, 'shear_modulus': 80},
    'Al': {'atomic_weight': 26.9815386, 'density': 2.7, 'heat_capacity': 24.2, 'thermal_conductivity': 235, 'atomic_radius': 143, 'electronegativity_P': 1.61, 'electronegativity_A': 1.47, 'VEC': 3, 'melting_point': 933.47, 'shear_modulus': 26},
    'Cu': {'atomic_weight': 63.546, 'density': 8.96, 'heat_capacity': 24.44, 'thermal_conductivity': 398, 'atomic_radius': 128, 'electronegativity_P': 1.9, 'electronegativity_A': 1.75, 'VEC': 11, 'melting_point': 1357.77, 'shear_modulus': 48},
    'Ti': {'atomic_weight': 47.867, 'density': 4.54, 'heat_capacity': 25.06, 'thermal_conductivity': 21.9, 'atomic_radius': 147, 'electronegativity_P': 1.54, 'electronegativity_A': 1.46, 'VEC': 4, 'melting_point': 1941, 'shear_modulus': 44},
    'Zr': {'atomic_weight': 91.224, 'density': 6.52, 'heat_capacity': 25.36, 'thermal_conductivity': 22.7, 'atomic_radius': 155, 'electronegativity_P': 1.33, 'electronegativity_A': 1.33, 'VEC': 4, 'melting_point': 2128, 'shear_modulus': 33},
    'Nb': {'atomic_weight': 92.90637, 'density': 8.57, 'heat_capacity': 24.6, 'thermal_conductivity': 53.7, 'atomic_radius': 146, 'electronegativity_P': 1.6, 'electronegativity_A': 1.6, 'VEC': 5, 'melting_point': 2750, 'shear_modulus': 38},
    'V': {'atomic_weight': 50.9415, 'density': 6.11, 'heat_capacity': 24.89, 'thermal_conductivity': 30.7, 'atomic_radius': 134, 'electronegativity_P': 1.63, 'electronegativity_A': 1.63, 'VEC': 5, 'melting_point': 2183, 'shear_modulus': 47},
    'Mo': {'atomic_weight': 95.95, 'density': 10.28, 'heat_capacity': 24.06, 'thermal_conductivity': 138, 'atomic_radius': 139, 'electronegativity_P': 2.16, 'electronegativity_A': 2.16, 'VEC': 6, 'melting_point': 2896, 'shear_modulus': 120},
    'Hf': {'atomic_weight': 178.49, 'density': 13.31, 'heat_capacity': 25.73, 'thermal_conductivity': 23.0, 'atomic_radius': 159, 'electronegativity_P': 1.3, 'electronegativity_A': 1.3, 'VEC': 4, 'melting_point': 2506, 'shear_modulus': 30},
    'Ta': {'atomic_weight': 180.94788, 'density': 16.69, 'heat_capacity': 25.36, 'thermal_conductivity': 57.5, 'atomic_radius': 146, 'electronegativity_P': 1.5, 'electronegativity_A': 1.5, 'VEC': 5, 'melting_point': 3290, 'shear_modulus': 69},
    'Si': {'atomic_weight': 28.085, 'density': 2.33, 'heat_capacity': 19.79, 'thermal_conductivity': 149, 'atomic_radius': 117, 'electronegativity_P': 1.9, 'electronegativity_A': 1.74, 'VEC': 4, 'melting_point': 1687, 'shear_modulus': 60},
    'W': {'atomic_weight': 183.84, 'density': 19.25, 'heat_capacity': 24.27, 'thermal_conductivity': 173, 'atomic_radius': 139, 'electronegativity_P': 2.36, 'electronegativity_A': 2.36, 'VEC': 6, 'melting_point': 3695, 'shear_modulus': 161},
}

# Binary mixed enthalpy
full_mixing_enthalpy_dict = {
    frozenset({'Co', 'Fe'}): -1,
    frozenset({'Ni', 'Fe'}): -2,
    frozenset({'Cr', 'Fe'}): -1,
    frozenset({'Mn', 'Fe'}): 0,
    frozenset({'Al', 'Fe'}): -11,
    frozenset({'Cu', 'Fe'}): 13,
    frozenset({'Ti', 'Fe'}): -17,
    frozenset({'Zr', 'Fe'}): -25,
    frozenset({'Fe', 'Nb'}): -16,
    frozenset({'V', 'Fe'}): -7,
    frozenset({'Mo', 'Fe'}): -2,
    frozenset({'Hf', 'Fe'}): -21,
    frozenset({'Ta', 'Fe'}): -15,
    frozenset({'Fe', 'Si'}): -35,
    frozenset({'Fe', 'W'}): 0,
    frozenset({'Ni', 'Co'}): 0,
    frozenset({'Co', 'Cr'}): -4,
    frozenset({'Mn', 'Co'}): -5,
    frozenset({'Co', 'Al'}): -19,
    frozenset({'Cu', 'Co'}): 6,
    frozenset({'Ti', 'Co'}): -28,
    frozenset({'Zr', 'Co'}): -41,
    frozenset({'Co', 'Nb'}): -25,
    frozenset({'Co', 'V'}): -14,
    frozenset({'Mo', 'Co'}): -5,
    frozenset({'Hf', 'Co'}): -35,
    frozenset({'Ta', 'Co'}): -24,
    frozenset({'Co', 'Si'}): -38,
    frozenset({'Co', 'W'}): -1,
    frozenset({'Ni', 'Cr'}): -7,
    frozenset({'Mn', 'Ni'}): -8,
    frozenset({'Ni', 'Al'}): -22,
    frozenset({'Cu', 'Ni'}): 4,
    frozenset({'Ni', 'Ti'}): -35,
    frozenset({'Ni', 'Zr'}): -49,
    frozenset({'Ni', 'Nb'}): -30,
    frozenset({'Ni', 'V'}): -18,
    frozenset({'Mo', 'Ni'}): -7,
    frozenset({'Hf', 'Ni'}): -42,
    frozenset({'Ta', 'Ni'}): -29,
    frozenset({'Ni', 'Si'}): -40,
    frozenset({'Ni', 'W'}): -3,
    frozenset({'Mn', 'Cr'}): 2,
    frozenset({'Cr', 'Al'}): -10,
    frozenset({'Cu', 'Cr'}): 12,
    frozenset({'Ti', 'Cr'}): -7,
    frozenset({'Zr', 'Cr'}): -12,
    frozenset({'Cr', 'Nb'}): -7,
    frozenset({'Cr', 'V'}): -2,
    frozenset({'Mo', 'Cr'}): 0,
    frozenset({'Hf', 'Cr'}): -9,
    frozenset({'Ta', 'Cr'}): -7,
    frozenset({'Cr', 'Si'}): -37,
    frozenset({'Cr', 'W'}): 1,
    frozenset({'Mn', 'Al'}): -19,
    frozenset({'Cu', 'Mn'}): 4,
    frozenset({'Mn', 'Ti'}): -8,
    frozenset({'Mn', 'Zr'}): -15,
    frozenset({'Mn', 'Nb'}): -4,
    frozenset({'Mn', 'V'}): -1,
    frozenset({'Mo', 'Mn'}): 5,
    frozenset({'Hf', 'Mn'}): -12,
    frozenset({'Ta', 'Mn'}): -4,
    frozenset({'Mn', 'Si'}): -45,
    frozenset({'Mn', 'W'}): 6,
    frozenset({'Cu', 'Al'}): -1,
    frozenset({'Ti', 'Al'}): -30,
    frozenset({'Zr', 'Al'}): -44,
    frozenset({'Al', 'Nb'}): -18,
    frozenset({'Al', 'V'}): -16,
    frozenset({'Mo', 'Al'}): -5,
    frozenset({'Hf', 'Al'}): -39,
    frozenset({'Ta', 'Al'}): -19,
    frozenset({'Al', 'Si'}): -19,
    frozenset({'Al', 'W'}): -2,
    frozenset({'Cu', 'Ti'}): -9,
    frozenset({'Cu', 'Zr'}): -23,
    frozenset({'Cu', 'Nb'}): 3,
    frozenset({'Cu', 'V'}): 5,
    frozenset({'Cu', 'Mo'}): 19,
    frozenset({'Cu', 'Hf'}): -17,
    frozenset({'Cu', 'Ta'}): 2,
    frozenset({'Cu', 'Si'}): -19,
    frozenset({'Cu', 'W'}): 22,
    frozenset({'Ti', 'Zr'}): 0,
    frozenset({'Ti', 'Nb'}): 2,
    frozenset({'Ti', 'V'}): -2,
    frozenset({'Mo', 'Ti'}): -4,
    frozenset({'Hf', 'Ti'}): 0,
    frozenset({'Ta', 'Ti'}): 1,
    frozenset({'Ti', 'Si'}): -66,
    frozenset({'Ti', 'W'}): -6,
    frozenset({'Zr', 'Nb'}): 4,
    frozenset({'Zr', 'V'}): -4,
    frozenset({'Mo', 'Zr'}): -6,
    frozenset({'Hf', 'Zr'}): 0,
    frozenset({'Ta', 'Zr'}): 3,
    frozenset({'Zr', 'Si'}): -84,
    frozenset({'Zr', 'W'}): -9,
    frozenset({'V', 'Nb'}): -1,
    frozenset({'Mo', 'Nb'}): -6,
    frozenset({'Hf', 'Nb'}): 4,
    frozenset({'Ta', 'Nb'}): 0,
    frozenset({'Si', 'Nb'}): -56,
    frozenset({'W', 'Nb'}): -8,
    frozenset({'Mo', 'V'}): 0,
    frozenset({'Hf', 'V'}): -2,
    frozenset({'Ta', 'V'}): -1,
    frozenset({'V', 'Si'}): -48,
    frozenset({'V', 'W'}): -1,
    frozenset({'Mo', 'Hf'}): -4,
    frozenset({'Mo', 'Ta'}): -5,
    frozenset({'Mo', 'Si'}): -35,
    frozenset({'Mo', 'W'}): 0,
    frozenset({'Hf', 'Ta'}): 3,
    frozenset({'Hf', 'Si'}): -77,
    frozenset({'Hf', 'W'}): -6,
    frozenset({'Ta', 'Si'}): -56,
    frozenset({'Ta', 'W'}): -7,
    frozenset({'W', 'Si'}): -31,
}

# lattice constant
lattice_constants = {
    'Fe': {'structure': 'bcc', 'a': 2.8665},
    'Co': {'structure': 'hcp', 'a': 2.507, 'c': 4.069},
    'Ni': {'structure': 'fcc', 'a': 3.524},
    'Cr': {'structure': 'bcc', 'a': 2.91},
    'Mn': {'structure': 'cubic', 'a': 8.89},
    'Al': {'structure': 'fcc', 'a': 4.05},
    'Cu': {'structure': 'fcc', 'a': 3.615},
    'Ti': {'structure': 'hcp', 'a': 2.95, 'c': 4.68},
    'Zr': {'structure': 'hcp', 'a': 3.23, 'c': 5.15},
    'Nb': {'structure': 'bcc', 'a': 3.3},
    'V': {'structure': 'bcc', 'a': 3.03},
    'Mo': {'structure': 'bcc', 'a': 3.15},
    'Hf': {'structure': 'hcp', 'a': 3.2, 'c': 5.05},
    'Ta': {'structure': 'bcc', 'a': 3.3},
    'Si': {'structure': 'diamond', 'a': 5.43},
    'W': {'structure': 'bcc', 'a': 3.165},
}
for elem, lattice in lattice_constants.items():
    if 'c' in lattice:
        element_properties[elem]['lattice_constant'] = (lattice['a'] + lattice['c']) / 2
    else:
        element_properties[elem]['lattice_constant'] = lattice['a']

# Descriptor calculation formula
class DescriptorsFunctions:
    """
    Class encapsulation of descriptor computation functions.
    """

    @staticmethod
    def delta_chi_allen(C, chi):
        """
        Poor electronegativity of Allen
        """
        chi_mean = sum([C[i] * chi[i] for i in range(len(C))])
        return math.sqrt(sum([C[i] * (chi[i] - chi_mean)**2 for i in range(len(C))]))

    @staticmethod
    def delta_chi_pauling(C, chi):
        """
        Poor Bowling electronegativity
        """
        return DescriptorsFunctions.delta_chi_allen(C, chi)

    @staticmethod
    def delta_S_mix(C, R):
        """
        Mixed entropy
        """
        return -R * sum([C[i] * math.log(C[i]) for i in range(len(C))])

    @staticmethod
    def delta_H_mix(C, H):
        """
        Mixed enthalpy
        """
        return sum([4 * H[i][j] * C[i] * C[j] for i in range(len(C)) for j in range(len(C)) if i != j])

    @staticmethod
    def delta_radii(C, r):
        """
        Atomic size difference
        """
        r_mean = sum([C[i] * r[i] for i in range(len(C))])
        return math.sqrt(sum([C[i] * (1 - r[i] / r_mean)**2 for i in range(len(C))]))

    @staticmethod
    def delta_a(C, a):
        """
        Difference in lattice constant
        """
        a_mean = sum([C[i] * a[i] for i in range(len(C))])
        return math.sqrt(sum([C[i] * (a[i] - a_mean)**2 for i in range(len(C))]))

    @staticmethod
    def delta_Tm(C, Tm):
        """
        Melting point difference
        """
        Tm_mean = sum([C[i] * Tm[i] for i in range(len(C))])
        return math.sqrt(sum([C[i] * (Tm[i] - Tm_mean)**2 for i in range(len(C))]))

    @staticmethod
    def lambda_param(delta_S_mix, delta_radii):
        """
        Geometric parameter
        """
        return delta_S_mix / delta_radii**2

    @staticmethod
    def Omega(Tm_mean, delta_S_mix, delta_H_mix):
        """
        Solid State Formation Parameters
        """
        if delta_H_mix == 0 or (Tm_mean * delta_S_mix / abs(delta_H_mix) > 100000):
            return 100000
        return Tm_mean * delta_S_mix / abs(delta_H_mix)

    @staticmethod
    def Tm_mixture(C, Tm):
        """
        Average melting temperature
        """
        return sum([C[i] * Tm[i] for i in range(len(C))])

    @staticmethod
    def am_mixture(C, a):
        """
        Average lattice constant
        """
        return sum([C[i] * a[i] for i in range(len(C))])

    @staticmethod
    def VEC_mixture(C, VEC):
        """
        Average valence electron
        """
        return sum([C[i] * VEC[i] for i in range(len(C))])

    @staticmethod
    def delta_G(C, G):
        """
        Shear modulus difference
        """
        G_mean = sum([C[i] * G[i] for i in range(len(C))])
        return math.sqrt(sum([C[i] * (G[i] - G_mean)**2 for i in range(len(C))]))

    @staticmethod
    def Gm_mixture(C, G):
        """
        Average shear modulus
        """
        return sum([C[i] * G[i] for i in range(len(C))])

# 描述符计算
class DescriptorsCalculator:
    """
    Descriptor calculator class
    """

    def __init__(self):
        pass

    def compute_descriptors(self, element_symbols, atomic_percentages):
        """
        Calculates a descriptor for a given element symbol and atomic percentage.

        :parameter
            element_symbols: A list of element symbols.
            atomic_percentages: A list of atomic percentages corresponding to element symbols.

        :return
            A dictionary containing the computed descriptors.
        """
        # Filtering non-zero percentage elements
        filtered_elements = [element for element, perc in zip(element_symbols, atomic_percentages) if perc > 0]
        filtered_percentages = [perc for perc in atomic_percentages if perc > 0]

        # Normalization ensures that the sum is 1
        total_percentage = sum(filtered_percentages)
        filtered_percentages = [perc / total_percentage for perc in filtered_percentages]

        C = filtered_percentages
        chi_allen = [element_properties[element]['electronegativity_A'] for element in filtered_elements]
        chi_pauling = [element_properties[element]['electronegativity_P'] for element in filtered_elements]
        R = 8.314  # General gas constant in J/mol-K

        H = [
            [full_mixing_enthalpy_dict[frozenset([filtered_elements[i], filtered_elements[j]])]
             if i != j and frozenset(
                [filtered_elements[i], filtered_elements[j]]) in full_mixing_enthalpy_dict else None
             for j in range(len(filtered_elements))]
            for i in range(len(filtered_elements))
        ]

        r = [element_properties[element]['atomic_radius'] for element in filtered_elements]
        a = [lattice_constants[element]['a'] for element in filtered_elements]
        Tm = [element_properties[element]['melting_point'] for element in filtered_elements]
        VEC = [element_properties[element]['VEC'] for element in filtered_elements]
        G = [element_properties[element]['shear_modulus'] for element in filtered_elements]

        # Descriptor calculation
        descriptors = {
            'delta_chi_allen': DescriptorsFunctions.delta_chi_allen(C, chi_allen),
            'delta_chi_pauling': DescriptorsFunctions.delta_chi_pauling(C, chi_pauling),
            'delta_S_mix': DescriptorsFunctions.delta_S_mix(C, R),
            'delta_H_mix': DescriptorsFunctions.delta_H_mix(C, H),
            'delta_radii': DescriptorsFunctions.delta_radii(C, r),
            'delta_a': DescriptorsFunctions.delta_a(C, a),
            'delta_Tm': DescriptorsFunctions.delta_Tm(C, Tm),
            'Tm_mean': DescriptorsFunctions.Tm_mixture(C, Tm),
            'am_mean': DescriptorsFunctions.am_mixture(C, a),
            'VEC_mean': DescriptorsFunctions.VEC_mixture(C, VEC),
            'delta_G': DescriptorsFunctions.delta_G(C, G),
            'Gm_mean': DescriptorsFunctions.Gm_mixture(C, G),
            'lambda': DescriptorsFunctions.lambda_param(DescriptorsFunctions.delta_S_mix(C, R),
                                                        DescriptorsFunctions.delta_radii(C, r)),
            'Omega': DescriptorsFunctions.Omega(DescriptorsFunctions.Tm_mixture(C, Tm),
                                                DescriptorsFunctions.delta_S_mix(C, R),
                                                DescriptorsFunctions.delta_H_mix(C, H))
        }

        return descriptors

