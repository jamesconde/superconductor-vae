"""
Element properties database for superconductor encoding.

Contains periodic table properties used for feature engineering:
- Electronegativity (Pauling scale)
- Atomic radius (pm)
- Ionization energy (kJ/mol)
- Electron affinity (kJ/mol)
- Melting point (K)
- Density (g/cm³)
- Thermal conductivity (W/m·K)
- Valence electrons
- d-electron count
- f-electron count
- Atomic mass (amu)
- Common oxidation states
"""

import numpy as np
from typing import Dict, List, Optional, Any


# Element symbols indexed by atomic number (1-indexed, index 0 is placeholder)
ELEMENT_SYMBOLS = [
    '',  # Placeholder for index 0
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

SYMBOL_TO_Z = {symbol: i for i, symbol in enumerate(ELEMENT_SYMBOLS) if symbol}

# Comprehensive element properties dictionary
# Key properties for superconductor Tc prediction
ELEMENT_PROPERTIES: Dict[str, Dict[str, Any]] = {
    # Period 1
    'H': {'Z': 1, 'mass': 1.008, 'electronegativity': 2.20, 'atomic_radius': 53,
          'ionization_energy': 1312, 'electron_affinity': 73, 'melting_point': 14,
          'density': 0.09, 'thermal_conductivity': 0.18, 'valence': 1, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [-1, 1]},
    'He': {'Z': 2, 'mass': 4.003, 'electronegativity': 0, 'atomic_radius': 31,
           'ionization_energy': 2372, 'electron_affinity': 0, 'melting_point': 1,
           'density': 0.18, 'thermal_conductivity': 0.15, 'valence': 0, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [0]},

    # Period 2
    'Li': {'Z': 3, 'mass': 6.94, 'electronegativity': 0.98, 'atomic_radius': 167,
           'ionization_energy': 520, 'electron_affinity': 60, 'melting_point': 454,
           'density': 0.53, 'thermal_conductivity': 85, 'valence': 1, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [1]},
    'Be': {'Z': 4, 'mass': 9.012, 'electronegativity': 1.57, 'atomic_radius': 112,
           'ionization_energy': 900, 'electron_affinity': 0, 'melting_point': 1560,
           'density': 1.85, 'thermal_conductivity': 190, 'valence': 2, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [2]},
    'B': {'Z': 5, 'mass': 10.81, 'electronegativity': 2.04, 'atomic_radius': 87,
          'ionization_energy': 801, 'electron_affinity': 27, 'melting_point': 2349,
          'density': 2.34, 'thermal_conductivity': 27, 'valence': 3, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [3]},
    'C': {'Z': 6, 'mass': 12.01, 'electronegativity': 2.55, 'atomic_radius': 67,
          'ionization_energy': 1086, 'electron_affinity': 122, 'melting_point': 3823,
          'density': 2.27, 'thermal_conductivity': 140, 'valence': 4, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [-4, -3, -2, -1, 1, 2, 3, 4]},
    'N': {'Z': 7, 'mass': 14.01, 'electronegativity': 3.04, 'atomic_radius': 56,
          'ionization_energy': 1402, 'electron_affinity': 0, 'melting_point': 63,
          'density': 1.25, 'thermal_conductivity': 0.026, 'valence': 5, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [-3, 3, 5]},
    'O': {'Z': 8, 'mass': 16.00, 'electronegativity': 3.44, 'atomic_radius': 48,
          'ionization_energy': 1314, 'electron_affinity': 141, 'melting_point': 55,
          'density': 1.43, 'thermal_conductivity': 0.027, 'valence': 6, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [-2, -1]},
    'F': {'Z': 9, 'mass': 19.00, 'electronegativity': 3.98, 'atomic_radius': 42,
          'ionization_energy': 1681, 'electron_affinity': 328, 'melting_point': 54,
          'density': 1.70, 'thermal_conductivity': 0.028, 'valence': 7, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [-1]},
    'Ne': {'Z': 10, 'mass': 20.18, 'electronegativity': 0, 'atomic_radius': 38,
           'ionization_energy': 2081, 'electron_affinity': 0, 'melting_point': 25,
           'density': 0.90, 'thermal_conductivity': 0.049, 'valence': 0, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [0]},

    # Period 3
    'Na': {'Z': 11, 'mass': 22.99, 'electronegativity': 0.93, 'atomic_radius': 190,
           'ionization_energy': 496, 'electron_affinity': 53, 'melting_point': 371,
           'density': 0.97, 'thermal_conductivity': 140, 'valence': 1, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [1]},
    'Mg': {'Z': 12, 'mass': 24.31, 'electronegativity': 1.31, 'atomic_radius': 145,
           'ionization_energy': 738, 'electron_affinity': 0, 'melting_point': 923,
           'density': 1.74, 'thermal_conductivity': 160, 'valence': 2, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [2]},
    'Al': {'Z': 13, 'mass': 26.98, 'electronegativity': 1.61, 'atomic_radius': 118,
           'ionization_energy': 578, 'electron_affinity': 43, 'melting_point': 933,
           'density': 2.70, 'thermal_conductivity': 235, 'valence': 3, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [3]},
    'Si': {'Z': 14, 'mass': 28.09, 'electronegativity': 1.90, 'atomic_radius': 111,
           'ionization_energy': 786, 'electron_affinity': 134, 'melting_point': 1687,
           'density': 2.33, 'thermal_conductivity': 150, 'valence': 4, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [-4, 4]},
    'P': {'Z': 15, 'mass': 30.97, 'electronegativity': 2.19, 'atomic_radius': 98,
          'ionization_energy': 1012, 'electron_affinity': 72, 'melting_point': 317,
          'density': 1.82, 'thermal_conductivity': 0.24, 'valence': 5, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [-3, 3, 5]},
    'S': {'Z': 16, 'mass': 32.07, 'electronegativity': 2.58, 'atomic_radius': 88,
          'ionization_energy': 1000, 'electron_affinity': 200, 'melting_point': 388,
          'density': 2.07, 'thermal_conductivity': 0.27, 'valence': 6, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [-2, 4, 6]},
    'Cl': {'Z': 17, 'mass': 35.45, 'electronegativity': 3.16, 'atomic_radius': 79,
           'ionization_energy': 1251, 'electron_affinity': 349, 'melting_point': 172,
           'density': 3.21, 'thermal_conductivity': 0.0089, 'valence': 7, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [-1, 1, 3, 5, 7]},
    'Ar': {'Z': 18, 'mass': 39.95, 'electronegativity': 0, 'atomic_radius': 71,
           'ionization_energy': 1521, 'electron_affinity': 0, 'melting_point': 84,
           'density': 1.78, 'thermal_conductivity': 0.018, 'valence': 0, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [0]},

    # Period 4 - includes important superconductor elements
    'K': {'Z': 19, 'mass': 39.10, 'electronegativity': 0.82, 'atomic_radius': 243,
          'ionization_energy': 419, 'electron_affinity': 48, 'melting_point': 337,
          'density': 0.86, 'thermal_conductivity': 100, 'valence': 1, 'd_electrons': 0, 'f_electrons': 0,
          'oxidation_states': [1]},
    'Ca': {'Z': 20, 'mass': 40.08, 'electronegativity': 1.00, 'atomic_radius': 194,
           'ionization_energy': 590, 'electron_affinity': 2, 'melting_point': 1115,
           'density': 1.55, 'thermal_conductivity': 200, 'valence': 2, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [2]},
    'Sc': {'Z': 21, 'mass': 44.96, 'electronegativity': 1.36, 'atomic_radius': 184,
           'ionization_energy': 633, 'electron_affinity': 18, 'melting_point': 1814,
           'density': 2.99, 'thermal_conductivity': 16, 'valence': 3, 'd_electrons': 1, 'f_electrons': 0,
           'oxidation_states': [3]},
    'Ti': {'Z': 22, 'mass': 47.87, 'electronegativity': 1.54, 'atomic_radius': 176,
           'ionization_energy': 659, 'electron_affinity': 8, 'melting_point': 1941,
           'density': 4.51, 'thermal_conductivity': 22, 'valence': 4, 'd_electrons': 2, 'f_electrons': 0,
           'oxidation_states': [2, 3, 4]},
    'V': {'Z': 23, 'mass': 50.94, 'electronegativity': 1.63, 'atomic_radius': 171,
          'ionization_energy': 651, 'electron_affinity': 51, 'melting_point': 2183,
          'density': 6.11, 'thermal_conductivity': 31, 'valence': 5, 'd_electrons': 3, 'f_electrons': 0,
          'oxidation_states': [2, 3, 4, 5]},
    'Cr': {'Z': 24, 'mass': 52.00, 'electronegativity': 1.66, 'atomic_radius': 166,
           'ionization_energy': 653, 'electron_affinity': 64, 'melting_point': 2180,
           'density': 7.19, 'thermal_conductivity': 94, 'valence': 6, 'd_electrons': 5, 'f_electrons': 0,
           'oxidation_states': [2, 3, 6]},
    'Mn': {'Z': 25, 'mass': 54.94, 'electronegativity': 1.55, 'atomic_radius': 161,
           'ionization_energy': 717, 'electron_affinity': 0, 'melting_point': 1519,
           'density': 7.47, 'thermal_conductivity': 8, 'valence': 7, 'd_electrons': 5, 'f_electrons': 0,
           'oxidation_states': [2, 3, 4, 7]},
    'Fe': {'Z': 26, 'mass': 55.85, 'electronegativity': 1.83, 'atomic_radius': 156,
           'ionization_energy': 762, 'electron_affinity': 16, 'melting_point': 1811,
           'density': 7.87, 'thermal_conductivity': 80, 'valence': 8, 'd_electrons': 6, 'f_electrons': 0,
           'oxidation_states': [2, 3]},
    'Co': {'Z': 27, 'mass': 58.93, 'electronegativity': 1.88, 'atomic_radius': 152,
           'ionization_energy': 760, 'electron_affinity': 64, 'melting_point': 1768,
           'density': 8.90, 'thermal_conductivity': 100, 'valence': 9, 'd_electrons': 7, 'f_electrons': 0,
           'oxidation_states': [2, 3]},
    'Ni': {'Z': 28, 'mass': 58.69, 'electronegativity': 1.91, 'atomic_radius': 149,
           'ionization_energy': 737, 'electron_affinity': 112, 'melting_point': 1728,
           'density': 8.91, 'thermal_conductivity': 91, 'valence': 10, 'd_electrons': 8, 'f_electrons': 0,
           'oxidation_states': [2, 3]},
    'Cu': {'Z': 29, 'mass': 63.55, 'electronegativity': 1.90, 'atomic_radius': 145,
           'ionization_energy': 745, 'electron_affinity': 119, 'melting_point': 1358,
           'density': 8.96, 'thermal_conductivity': 400, 'valence': 11, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [1, 2]},  # Critical for cuprate superconductors
    'Zn': {'Z': 30, 'mass': 65.38, 'electronegativity': 1.65, 'atomic_radius': 142,
           'ionization_energy': 906, 'electron_affinity': 0, 'melting_point': 693,
           'density': 7.14, 'thermal_conductivity': 120, 'valence': 2, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [2]},
    'Ga': {'Z': 31, 'mass': 69.72, 'electronegativity': 1.81, 'atomic_radius': 136,
           'ionization_energy': 579, 'electron_affinity': 29, 'melting_point': 303,
           'density': 5.91, 'thermal_conductivity': 29, 'valence': 3, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [3]},
    'Ge': {'Z': 32, 'mass': 72.63, 'electronegativity': 2.01, 'atomic_radius': 125,
           'ionization_energy': 762, 'electron_affinity': 119, 'melting_point': 1211,
           'density': 5.32, 'thermal_conductivity': 60, 'valence': 4, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [2, 4]},
    'As': {'Z': 33, 'mass': 74.92, 'electronegativity': 2.18, 'atomic_radius': 114,
           'ionization_energy': 947, 'electron_affinity': 78, 'melting_point': 1090,
           'density': 5.73, 'thermal_conductivity': 50, 'valence': 5, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [-3, 3, 5]},  # Important for iron-based superconductors
    'Se': {'Z': 34, 'mass': 78.97, 'electronegativity': 2.55, 'atomic_radius': 103,
           'ionization_energy': 941, 'electron_affinity': 195, 'melting_point': 494,
           'density': 4.81, 'thermal_conductivity': 0.52, 'valence': 6, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [-2, 4, 6]},
    'Br': {'Z': 35, 'mass': 79.90, 'electronegativity': 2.96, 'atomic_radius': 94,
           'ionization_energy': 1140, 'electron_affinity': 325, 'melting_point': 266,
           'density': 3.12, 'thermal_conductivity': 0.12, 'valence': 7, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [-1, 1, 5]},
    'Kr': {'Z': 36, 'mass': 83.80, 'electronegativity': 3.00, 'atomic_radius': 88,
           'ionization_energy': 1351, 'electron_affinity': 0, 'melting_point': 116,
           'density': 3.75, 'thermal_conductivity': 0.0095, 'valence': 0, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [0, 2]},

    # Period 5 - includes important superconductor elements
    'Rb': {'Z': 37, 'mass': 85.47, 'electronegativity': 0.82, 'atomic_radius': 265,
           'ionization_energy': 403, 'electron_affinity': 47, 'melting_point': 312,
           'density': 1.53, 'thermal_conductivity': 58, 'valence': 1, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [1]},
    'Sr': {'Z': 38, 'mass': 87.62, 'electronegativity': 0.95, 'atomic_radius': 219,
           'ionization_energy': 550, 'electron_affinity': 5, 'melting_point': 1050,
           'density': 2.63, 'thermal_conductivity': 35, 'valence': 2, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [2]},  # Common in cuprate superconductors
    'Y': {'Z': 39, 'mass': 88.91, 'electronegativity': 1.22, 'atomic_radius': 212,
          'ionization_energy': 600, 'electron_affinity': 30, 'melting_point': 1799,
          'density': 4.47, 'thermal_conductivity': 17, 'valence': 3, 'd_electrons': 1, 'f_electrons': 0,
          'oxidation_states': [3]},  # YBCO superconductor
    'Zr': {'Z': 40, 'mass': 91.22, 'electronegativity': 1.33, 'atomic_radius': 206,
           'ionization_energy': 640, 'electron_affinity': 41, 'melting_point': 2128,
           'density': 6.51, 'thermal_conductivity': 23, 'valence': 4, 'd_electrons': 2, 'f_electrons': 0,
           'oxidation_states': [4]},
    'Nb': {'Z': 41, 'mass': 92.91, 'electronegativity': 1.60, 'atomic_radius': 198,
           'ionization_energy': 652, 'electron_affinity': 86, 'melting_point': 2750,
           'density': 8.57, 'thermal_conductivity': 54, 'valence': 5, 'd_electrons': 4, 'f_electrons': 0,
           'oxidation_states': [3, 5]},  # Nb3Sn, NbTi superconductors
    'Mo': {'Z': 42, 'mass': 95.95, 'electronegativity': 2.16, 'atomic_radius': 190,
           'ionization_energy': 684, 'electron_affinity': 72, 'melting_point': 2896,
           'density': 10.28, 'thermal_conductivity': 139, 'valence': 6, 'd_electrons': 5, 'f_electrons': 0,
           'oxidation_states': [4, 6]},
    'Tc': {'Z': 43, 'mass': 98.00, 'electronegativity': 1.90, 'atomic_radius': 183,
           'ionization_energy': 702, 'electron_affinity': 53, 'melting_point': 2430,
           'density': 11.50, 'thermal_conductivity': 51, 'valence': 7, 'd_electrons': 5, 'f_electrons': 0,
           'oxidation_states': [4, 7]},
    'Ru': {'Z': 44, 'mass': 101.07, 'electronegativity': 2.20, 'atomic_radius': 178,
           'ionization_energy': 710, 'electron_affinity': 101, 'melting_point': 2607,
           'density': 12.37, 'thermal_conductivity': 117, 'valence': 8, 'd_electrons': 7, 'f_electrons': 0,
           'oxidation_states': [3, 4]},
    'Rh': {'Z': 45, 'mass': 102.91, 'electronegativity': 2.28, 'atomic_radius': 173,
           'ionization_energy': 720, 'electron_affinity': 110, 'melting_point': 2237,
           'density': 12.45, 'thermal_conductivity': 150, 'valence': 9, 'd_electrons': 8, 'f_electrons': 0,
           'oxidation_states': [3]},
    'Pd': {'Z': 46, 'mass': 106.42, 'electronegativity': 2.20, 'atomic_radius': 169,
           'ionization_energy': 804, 'electron_affinity': 54, 'melting_point': 1828,
           'density': 12.02, 'thermal_conductivity': 72, 'valence': 10, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [2, 4]},
    'Ag': {'Z': 47, 'mass': 107.87, 'electronegativity': 1.93, 'atomic_radius': 165,
           'ionization_energy': 731, 'electron_affinity': 126, 'melting_point': 1235,
           'density': 10.49, 'thermal_conductivity': 429, 'valence': 11, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [1]},
    'Cd': {'Z': 48, 'mass': 112.41, 'electronegativity': 1.69, 'atomic_radius': 161,
           'ionization_energy': 868, 'electron_affinity': 0, 'melting_point': 594,
           'density': 8.65, 'thermal_conductivity': 97, 'valence': 2, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [2]},
    'In': {'Z': 49, 'mass': 114.82, 'electronegativity': 1.78, 'atomic_radius': 156,
           'ionization_energy': 558, 'electron_affinity': 29, 'melting_point': 430,
           'density': 7.31, 'thermal_conductivity': 82, 'valence': 3, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [3]},
    'Sn': {'Z': 50, 'mass': 118.71, 'electronegativity': 1.96, 'atomic_radius': 145,
           'ionization_energy': 709, 'electron_affinity': 107, 'melting_point': 505,
           'density': 7.31, 'thermal_conductivity': 67, 'valence': 4, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [2, 4]},  # Nb3Sn superconductor
    'Sb': {'Z': 51, 'mass': 121.76, 'electronegativity': 2.05, 'atomic_radius': 133,
           'ionization_energy': 834, 'electron_affinity': 103, 'melting_point': 904,
           'density': 6.70, 'thermal_conductivity': 24, 'valence': 5, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [-3, 3, 5]},
    'Te': {'Z': 52, 'mass': 127.60, 'electronegativity': 2.10, 'atomic_radius': 123,
           'ionization_energy': 869, 'electron_affinity': 190, 'melting_point': 723,
           'density': 6.24, 'thermal_conductivity': 3, 'valence': 6, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [-2, 4, 6]},
    'I': {'Z': 53, 'mass': 126.90, 'electronegativity': 2.66, 'atomic_radius': 115,
          'ionization_energy': 1008, 'electron_affinity': 295, 'melting_point': 387,
          'density': 4.94, 'thermal_conductivity': 0.45, 'valence': 7, 'd_electrons': 10, 'f_electrons': 0,
          'oxidation_states': [-1, 1, 5, 7]},
    'Xe': {'Z': 54, 'mass': 131.29, 'electronegativity': 2.60, 'atomic_radius': 108,
           'ionization_energy': 1170, 'electron_affinity': 0, 'melting_point': 161,
           'density': 5.90, 'thermal_conductivity': 0.0057, 'valence': 0, 'd_electrons': 10, 'f_electrons': 0,
           'oxidation_states': [0, 2, 4, 6]},

    # Period 6 - includes lanthanides important for superconductors
    'Cs': {'Z': 55, 'mass': 132.91, 'electronegativity': 0.79, 'atomic_radius': 298,
           'ionization_energy': 376, 'electron_affinity': 46, 'melting_point': 302,
           'density': 1.90, 'thermal_conductivity': 36, 'valence': 1, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [1]},
    'Ba': {'Z': 56, 'mass': 137.33, 'electronegativity': 0.89, 'atomic_radius': 253,
           'ionization_energy': 503, 'electron_affinity': 14, 'melting_point': 1000,
           'density': 3.51, 'thermal_conductivity': 18, 'valence': 2, 'd_electrons': 0, 'f_electrons': 0,
           'oxidation_states': [2]},  # YBCO, BSCCO superconductors
    'La': {'Z': 57, 'mass': 138.91, 'electronegativity': 1.10, 'atomic_radius': 250,
           'ionization_energy': 538, 'electron_affinity': 48, 'melting_point': 1193,
           'density': 6.16, 'thermal_conductivity': 13, 'valence': 3, 'd_electrons': 1, 'f_electrons': 0,
           'oxidation_states': [3]},  # La-based cuprates
    'Ce': {'Z': 58, 'mass': 140.12, 'electronegativity': 1.12, 'atomic_radius': 248,
           'ionization_energy': 534, 'electron_affinity': 50, 'melting_point': 1068,
           'density': 6.77, 'thermal_conductivity': 11, 'valence': 4, 'd_electrons': 1, 'f_electrons': 1,
           'oxidation_states': [3, 4]},
    'Pr': {'Z': 59, 'mass': 140.91, 'electronegativity': 1.13, 'atomic_radius': 247,
           'ionization_energy': 527, 'electron_affinity': 50, 'melting_point': 1208,
           'density': 6.77, 'thermal_conductivity': 13, 'valence': 5, 'd_electrons': 0, 'f_electrons': 3,
           'oxidation_states': [3, 4]},
    'Nd': {'Z': 60, 'mass': 144.24, 'electronegativity': 1.14, 'atomic_radius': 206,
           'ionization_energy': 533, 'electron_affinity': 50, 'melting_point': 1297,
           'density': 7.01, 'thermal_conductivity': 17, 'valence': 6, 'd_electrons': 0, 'f_electrons': 4,
           'oxidation_states': [3]},
    'Pm': {'Z': 61, 'mass': 145.00, 'electronegativity': 1.13, 'atomic_radius': 205,
           'ionization_energy': 540, 'electron_affinity': 50, 'melting_point': 1315,
           'density': 7.26, 'thermal_conductivity': 18, 'valence': 7, 'd_electrons': 0, 'f_electrons': 5,
           'oxidation_states': [3]},
    'Sm': {'Z': 62, 'mass': 150.36, 'electronegativity': 1.17, 'atomic_radius': 238,
           'ionization_energy': 545, 'electron_affinity': 50, 'melting_point': 1345,
           'density': 7.52, 'thermal_conductivity': 13, 'valence': 8, 'd_electrons': 0, 'f_electrons': 6,
           'oxidation_states': [2, 3]},
    'Eu': {'Z': 63, 'mass': 151.96, 'electronegativity': 1.20, 'atomic_radius': 231,
           'ionization_energy': 547, 'electron_affinity': 50, 'melting_point': 1099,
           'density': 5.24, 'thermal_conductivity': 14, 'valence': 9, 'd_electrons': 0, 'f_electrons': 7,
           'oxidation_states': [2, 3]},
    'Gd': {'Z': 64, 'mass': 157.25, 'electronegativity': 1.20, 'atomic_radius': 233,
           'ionization_energy': 593, 'electron_affinity': 50, 'melting_point': 1585,
           'density': 7.90, 'thermal_conductivity': 11, 'valence': 10, 'd_electrons': 1, 'f_electrons': 7,
           'oxidation_states': [3]},
    'Tb': {'Z': 65, 'mass': 158.93, 'electronegativity': 1.10, 'atomic_radius': 225,
           'ionization_energy': 566, 'electron_affinity': 50, 'melting_point': 1629,
           'density': 8.23, 'thermal_conductivity': 11, 'valence': 11, 'd_electrons': 0, 'f_electrons': 9,
           'oxidation_states': [3, 4]},
    'Dy': {'Z': 66, 'mass': 162.50, 'electronegativity': 1.22, 'atomic_radius': 228,
           'ionization_energy': 573, 'electron_affinity': 50, 'melting_point': 1680,
           'density': 8.54, 'thermal_conductivity': 11, 'valence': 12, 'd_electrons': 0, 'f_electrons': 10,
           'oxidation_states': [3]},
    'Ho': {'Z': 67, 'mass': 164.93, 'electronegativity': 1.23, 'atomic_radius': 226,
           'ionization_energy': 581, 'electron_affinity': 50, 'melting_point': 1734,
           'density': 8.80, 'thermal_conductivity': 16, 'valence': 13, 'd_electrons': 0, 'f_electrons': 11,
           'oxidation_states': [3]},
    'Er': {'Z': 68, 'mass': 167.26, 'electronegativity': 1.24, 'atomic_radius': 226,
           'ionization_energy': 589, 'electron_affinity': 50, 'melting_point': 1802,
           'density': 9.07, 'thermal_conductivity': 15, 'valence': 14, 'd_electrons': 0, 'f_electrons': 12,
           'oxidation_states': [3]},
    'Tm': {'Z': 69, 'mass': 168.93, 'electronegativity': 1.25, 'atomic_radius': 222,
           'ionization_energy': 597, 'electron_affinity': 50, 'melting_point': 1818,
           'density': 9.32, 'thermal_conductivity': 17, 'valence': 15, 'd_electrons': 0, 'f_electrons': 13,
           'oxidation_states': [2, 3]},
    'Yb': {'Z': 70, 'mass': 173.05, 'electronegativity': 1.10, 'atomic_radius': 222,
           'ionization_energy': 603, 'electron_affinity': 50, 'melting_point': 1097,
           'density': 6.90, 'thermal_conductivity': 39, 'valence': 16, 'd_electrons': 0, 'f_electrons': 14,
           'oxidation_states': [2, 3]},
    'Lu': {'Z': 71, 'mass': 174.97, 'electronegativity': 1.27, 'atomic_radius': 217,
           'ionization_energy': 524, 'electron_affinity': 50, 'melting_point': 1925,
           'density': 9.84, 'thermal_conductivity': 16, 'valence': 3, 'd_electrons': 1, 'f_electrons': 14,
           'oxidation_states': [3]},
    'Hf': {'Z': 72, 'mass': 178.49, 'electronegativity': 1.30, 'atomic_radius': 208,
           'ionization_energy': 659, 'electron_affinity': 0, 'melting_point': 2506,
           'density': 13.31, 'thermal_conductivity': 23, 'valence': 4, 'd_electrons': 2, 'f_electrons': 14,
           'oxidation_states': [4]},
    'Ta': {'Z': 73, 'mass': 180.95, 'electronegativity': 1.50, 'atomic_radius': 200,
           'ionization_energy': 761, 'electron_affinity': 31, 'melting_point': 3290,
           'density': 16.65, 'thermal_conductivity': 57, 'valence': 5, 'd_electrons': 3, 'f_electrons': 14,
           'oxidation_states': [5]},
    'W': {'Z': 74, 'mass': 183.84, 'electronegativity': 2.36, 'atomic_radius': 193,
          'ionization_energy': 770, 'electron_affinity': 79, 'melting_point': 3695,
          'density': 19.25, 'thermal_conductivity': 170, 'valence': 6, 'd_electrons': 4, 'f_electrons': 14,
          'oxidation_states': [4, 6]},
    'Re': {'Z': 75, 'mass': 186.21, 'electronegativity': 1.90, 'atomic_radius': 188,
           'ionization_energy': 760, 'electron_affinity': 14, 'melting_point': 3459,
           'density': 21.02, 'thermal_conductivity': 48, 'valence': 7, 'd_electrons': 5, 'f_electrons': 14,
           'oxidation_states': [4, 7]},
    'Os': {'Z': 76, 'mass': 190.23, 'electronegativity': 2.20, 'atomic_radius': 185,
           'ionization_energy': 840, 'electron_affinity': 106, 'melting_point': 3306,
           'density': 22.59, 'thermal_conductivity': 88, 'valence': 8, 'd_electrons': 6, 'f_electrons': 14,
           'oxidation_states': [4, 8]},
    'Ir': {'Z': 77, 'mass': 192.22, 'electronegativity': 2.20, 'atomic_radius': 180,
           'ionization_energy': 880, 'electron_affinity': 151, 'melting_point': 2719,
           'density': 22.56, 'thermal_conductivity': 147, 'valence': 9, 'd_electrons': 7, 'f_electrons': 14,
           'oxidation_states': [3, 4]},
    'Pt': {'Z': 78, 'mass': 195.08, 'electronegativity': 2.28, 'atomic_radius': 177,
           'ionization_energy': 870, 'electron_affinity': 205, 'melting_point': 2041,
           'density': 21.45, 'thermal_conductivity': 72, 'valence': 10, 'd_electrons': 9, 'f_electrons': 14,
           'oxidation_states': [2, 4]},
    'Au': {'Z': 79, 'mass': 196.97, 'electronegativity': 2.54, 'atomic_radius': 174,
           'ionization_energy': 890, 'electron_affinity': 223, 'melting_point': 1337,
           'density': 19.30, 'thermal_conductivity': 320, 'valence': 11, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [1, 3]},
    'Hg': {'Z': 80, 'mass': 200.59, 'electronegativity': 2.00, 'atomic_radius': 171,
           'ionization_energy': 1007, 'electron_affinity': 0, 'melting_point': 234,
           'density': 13.55, 'thermal_conductivity': 8, 'valence': 2, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [1, 2]},  # First superconductor discovered!
    'Tl': {'Z': 81, 'mass': 204.38, 'electronegativity': 1.62, 'atomic_radius': 156,
           'ionization_energy': 589, 'electron_affinity': 19, 'melting_point': 577,
           'density': 11.85, 'thermal_conductivity': 46, 'valence': 3, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [1, 3]},  # Tl-based cuprate superconductors
    'Pb': {'Z': 82, 'mass': 207.20, 'electronegativity': 2.33, 'atomic_radius': 154,
           'ionization_energy': 716, 'electron_affinity': 35, 'melting_point': 601,
           'density': 11.34, 'thermal_conductivity': 35, 'valence': 4, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [2, 4]},  # Classic superconductor (Tc = 7.2 K)
    'Bi': {'Z': 83, 'mass': 208.98, 'electronegativity': 2.02, 'atomic_radius': 143,
           'ionization_energy': 703, 'electron_affinity': 91, 'melting_point': 545,
           'density': 9.78, 'thermal_conductivity': 8, 'valence': 5, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [3, 5]},  # BSCCO superconductors
    'Po': {'Z': 84, 'mass': 209.00, 'electronegativity': 2.00, 'atomic_radius': 135,
           'ionization_energy': 812, 'electron_affinity': 183, 'melting_point': 527,
           'density': 9.20, 'thermal_conductivity': 20, 'valence': 6, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [2, 4]},
    'At': {'Z': 85, 'mass': 210.00, 'electronegativity': 2.20, 'atomic_radius': 127,
           'ionization_energy': 920, 'electron_affinity': 270, 'melting_point': 575,
           'density': 7.00, 'thermal_conductivity': 2, 'valence': 7, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [-1, 1]},
    'Rn': {'Z': 86, 'mass': 222.00, 'electronegativity': 0, 'atomic_radius': 120,
           'ionization_energy': 1037, 'electron_affinity': 0, 'melting_point': 202,
           'density': 9.73, 'thermal_conductivity': 0.0036, 'valence': 0, 'd_electrons': 10, 'f_electrons': 14,
           'oxidation_states': [0]},

    # Period 7 - actinides
    'Fr': {'Z': 87, 'mass': 223.00, 'electronegativity': 0.70, 'atomic_radius': 348,
           'ionization_energy': 380, 'electron_affinity': 47, 'melting_point': 300,
           'density': 1.87, 'thermal_conductivity': 15, 'valence': 1, 'd_electrons': 0, 'f_electrons': 14,
           'oxidation_states': [1]},
    'Ra': {'Z': 88, 'mass': 226.00, 'electronegativity': 0.90, 'atomic_radius': 283,
           'ionization_energy': 509, 'electron_affinity': 10, 'melting_point': 973,
           'density': 5.50, 'thermal_conductivity': 19, 'valence': 2, 'd_electrons': 0, 'f_electrons': 14,
           'oxidation_states': [2]},
    'Ac': {'Z': 89, 'mass': 227.00, 'electronegativity': 1.10, 'atomic_radius': 260,
           'ionization_energy': 499, 'electron_affinity': 33, 'melting_point': 1323,
           'density': 10.07, 'thermal_conductivity': 12, 'valence': 3, 'd_electrons': 1, 'f_electrons': 14,
           'oxidation_states': [3]},
    'Th': {'Z': 90, 'mass': 232.04, 'electronegativity': 1.30, 'atomic_radius': 237,
           'ionization_energy': 587, 'electron_affinity': 113, 'melting_point': 2115,
           'density': 11.72, 'thermal_conductivity': 54, 'valence': 4, 'd_electrons': 2, 'f_electrons': 14,
           'oxidation_states': [4]},
    'Pa': {'Z': 91, 'mass': 231.04, 'electronegativity': 1.50, 'atomic_radius': 243,
           'ionization_energy': 568, 'electron_affinity': 0, 'melting_point': 1841,
           'density': 15.37, 'thermal_conductivity': 47, 'valence': 5, 'd_electrons': 1, 'f_electrons': 2,
           'oxidation_states': [4, 5]},
    'U': {'Z': 92, 'mass': 238.03, 'electronegativity': 1.38, 'atomic_radius': 240,
          'ionization_energy': 598, 'electron_affinity': 0, 'melting_point': 1405,
          'density': 19.05, 'thermal_conductivity': 27, 'valence': 6, 'd_electrons': 1, 'f_electrons': 3,
          'oxidation_states': [3, 4, 5, 6]},
}


# Property keys for feature extraction
PROPERTY_KEYS = [
    'mass',
    'electronegativity',
    'atomic_radius',
    'ionization_energy',
    'electron_affinity',
    'melting_point',
    'density',
    'thermal_conductivity',
    'valence',
    'd_electrons',
    'f_electrons',
]


def get_element_property(symbol: str, property_name: str) -> Optional[float]:
    """
    Get a specific property for an element.

    Args:
        symbol: Element symbol (e.g., 'Cu', 'O')
        property_name: Property key (e.g., 'electronegativity')

    Returns:
        Property value or None if not found
    """
    if symbol not in ELEMENT_PROPERTIES:
        return None
    return ELEMENT_PROPERTIES[symbol].get(property_name)


def get_atomic_number(symbol: str) -> Optional[int]:
    """Get atomic number for element symbol."""
    return SYMBOL_TO_Z.get(symbol)


def get_element_symbol(z: int) -> Optional[str]:
    """Get element symbol for atomic number."""
    if 1 <= z < len(ELEMENT_SYMBOLS):
        return ELEMENT_SYMBOLS[z]
    return None


def get_all_properties(symbol: str) -> Optional[Dict[str, Any]]:
    """Get all properties for an element."""
    return ELEMENT_PROPERTIES.get(symbol)


def get_oxidation_states(symbol: str) -> List[int]:
    """Get common oxidation states for an element."""
    props = ELEMENT_PROPERTIES.get(symbol)
    if props:
        return props.get('oxidation_states', [])
    return []


# Superconductor-relevant element categories
SUPERCONDUCTOR_ELEMENTS = {
    'cuprate_transition_metals': ['Cu'],
    'cuprate_alkaline_earth': ['Ca', 'Sr', 'Ba'],
    'cuprate_rare_earth': ['Y', 'La', 'Nd', 'Sm', 'Eu', 'Gd'],
    'cuprate_post_transition': ['Tl', 'Pb', 'Bi'],
    'iron_based_transition': ['Fe', 'Co', 'Ni'],
    'iron_based_pnictogen': ['As', 'P'],
    'iron_based_chalcogen': ['Se', 'Te', 'S'],
    'conventional_type_I': ['Al', 'In', 'Sn', 'Pb', 'Hg'],
    'conventional_type_II': ['Nb', 'V', 'Ta', 'Ti'],
    'MgB2_type': ['Mg', 'B'],
    'hydride_host': ['H', 'La', 'Y', 'Ca', 'S'],  # High-pressure hydrides
}


def is_superconductor_relevant(symbol: str) -> bool:
    """Check if element is commonly found in superconductors."""
    for category, elements in SUPERCONDUCTOR_ELEMENTS.items():
        if symbol in elements:
            return True
    return False


def get_element_category(symbol: str) -> Optional[str]:
    """Get superconductor category for an element."""
    for category, elements in SUPERCONDUCTOR_ELEMENTS.items():
        if symbol in elements:
            return category
    return None
