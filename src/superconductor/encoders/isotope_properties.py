"""
Isotope properties database for superconductor encoding.

Includes:
- All stable isotopes for elements 1-92
- Key radioactive isotopes used in research
- Nuclear properties (mass, spin, abundance)
- Isotope effect parameters for superconductivity

The isotope effect is a key signature of phonon-mediated superconductivity:
    Tc ∝ M^(-α)  where α ≈ 0.5 for BCS superconductors

For high-Tc and unconventional superconductors, α can be anomalous,
providing insights into pairing mechanisms.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class IsotopeData:
    """Data for a single isotope."""
    symbol: str           # Element symbol
    mass_number: int      # A (protons + neutrons)
    atomic_mass: float    # Exact atomic mass (u)
    natural_abundance: float  # Natural abundance (0-1), 0 for synthetic
    nuclear_spin: float   # Nuclear spin quantum number
    is_stable: bool       # Whether isotope is stable
    half_life: Optional[float] = None  # Half-life in seconds (None if stable)

    @property
    def notation(self) -> str:
        """Get isotope notation like '2H' or '18O'."""
        return f"{self.mass_number}{self.symbol}"

    @property
    def is_naturally_occurring(self) -> bool:
        """Check if isotope occurs naturally."""
        return self.natural_abundance > 0


# Comprehensive isotope database
# Format: {element_symbol: {mass_number: IsotopeData}}
ISOTOPE_DATABASE: Dict[str, Dict[int, IsotopeData]] = {}


def _add_isotope(symbol: str, mass_number: int, atomic_mass: float,
                 abundance: float, spin: float, stable: bool,
                 half_life: Optional[float] = None):
    """Helper to add isotope to database."""
    if symbol not in ISOTOPE_DATABASE:
        ISOTOPE_DATABASE[symbol] = {}

    ISOTOPE_DATABASE[symbol][mass_number] = IsotopeData(
        symbol=symbol,
        mass_number=mass_number,
        atomic_mass=atomic_mass,
        natural_abundance=abundance,
        nuclear_spin=spin,
        is_stable=stable,
        half_life=half_life
    )


# =============================================================================
# Hydrogen - Critical for hydride superconductors
# =============================================================================
_add_isotope('H', 1, 1.00783, 0.99985, 0.5, True)   # Protium
_add_isotope('H', 2, 2.01410, 0.00015, 1.0, True)   # Deuterium (D)
_add_isotope('H', 3, 3.01605, 0.0, 0.5, False, 3.89e8)  # Tritium (T)

# =============================================================================
# Helium
# =============================================================================
_add_isotope('He', 3, 3.01603, 0.000001, 0.5, True)
_add_isotope('He', 4, 4.00260, 0.999999, 0.0, True)

# =============================================================================
# Lithium - Used in some exotic superconductors
# =============================================================================
_add_isotope('Li', 6, 6.01512, 0.0759, 1.0, True)
_add_isotope('Li', 7, 7.01600, 0.9241, 1.5, True)

# =============================================================================
# Beryllium
# =============================================================================
_add_isotope('Be', 9, 9.01218, 1.0, 1.5, True)

# =============================================================================
# Boron - Critical for MgB2 (isotope effect confirmed phonon mechanism)
# =============================================================================
_add_isotope('B', 10, 10.01294, 0.199, 3.0, True)
_add_isotope('B', 11, 11.00931, 0.801, 1.5, True)

# =============================================================================
# Carbon
# =============================================================================
_add_isotope('C', 12, 12.00000, 0.9893, 0.0, True)
_add_isotope('C', 13, 13.00335, 0.0107, 0.5, True)
_add_isotope('C', 14, 14.00324, 0.0, 0.0, False, 1.81e11)

# =============================================================================
# Nitrogen
# =============================================================================
_add_isotope('N', 14, 14.00307, 0.9963, 1.0, True)
_add_isotope('N', 15, 15.00011, 0.0037, 0.5, True)

# =============================================================================
# Oxygen - Critical for cuprate superconductors (O-18 substitution studies)
# =============================================================================
_add_isotope('O', 16, 15.99491, 0.99757, 0.0, True)
_add_isotope('O', 17, 16.99913, 0.00038, 2.5, True)
_add_isotope('O', 18, 17.99916, 0.00205, 0.0, True)

# =============================================================================
# Fluorine
# =============================================================================
_add_isotope('F', 19, 18.99840, 1.0, 0.5, True)

# =============================================================================
# Neon
# =============================================================================
_add_isotope('Ne', 20, 19.99244, 0.9048, 0.0, True)
_add_isotope('Ne', 21, 20.99385, 0.0027, 1.5, True)
_add_isotope('Ne', 22, 21.99138, 0.0925, 0.0, True)

# =============================================================================
# Sodium
# =============================================================================
_add_isotope('Na', 23, 22.98977, 1.0, 1.5, True)

# =============================================================================
# Magnesium - MgB2 superconductor
# =============================================================================
_add_isotope('Mg', 24, 23.98504, 0.7899, 0.0, True)
_add_isotope('Mg', 25, 24.98584, 0.1000, 2.5, True)
_add_isotope('Mg', 26, 25.98259, 0.1101, 0.0, True)

# =============================================================================
# Aluminum - Conventional superconductor
# =============================================================================
_add_isotope('Al', 27, 26.98154, 1.0, 2.5, True)

# =============================================================================
# Silicon
# =============================================================================
_add_isotope('Si', 28, 27.97693, 0.9223, 0.0, True)
_add_isotope('Si', 29, 28.97649, 0.0467, 0.5, True)
_add_isotope('Si', 30, 29.97377, 0.0310, 0.0, True)

# =============================================================================
# Phosphorus
# =============================================================================
_add_isotope('P', 31, 30.97376, 1.0, 0.5, True)

# =============================================================================
# Sulfur
# =============================================================================
_add_isotope('S', 32, 31.97207, 0.9499, 0.0, True)
_add_isotope('S', 33, 32.97146, 0.0075, 1.5, True)
_add_isotope('S', 34, 33.96787, 0.0425, 0.0, True)
_add_isotope('S', 36, 35.96708, 0.0001, 0.0, True)

# =============================================================================
# Chlorine
# =============================================================================
_add_isotope('Cl', 35, 34.96885, 0.7576, 1.5, True)
_add_isotope('Cl', 37, 36.96590, 0.2424, 1.5, True)

# =============================================================================
# Argon
# =============================================================================
_add_isotope('Ar', 36, 35.96755, 0.00334, 0.0, True)
_add_isotope('Ar', 38, 37.96273, 0.00063, 0.0, True)
_add_isotope('Ar', 40, 39.96238, 0.99603, 0.0, True)

# =============================================================================
# Potassium - K3C60 superconductor
# =============================================================================
_add_isotope('K', 39, 38.96371, 0.9326, 1.5, True)
_add_isotope('K', 40, 39.96400, 0.0001, 4.0, False, 4.0e16)  # Very long-lived
_add_isotope('K', 41, 40.96183, 0.0673, 1.5, True)

# =============================================================================
# Calcium - Cuprates, CaC6, hydrides
# =============================================================================
_add_isotope('Ca', 40, 39.96259, 0.96941, 0.0, True)
_add_isotope('Ca', 42, 41.95862, 0.00647, 0.0, True)
_add_isotope('Ca', 43, 42.95877, 0.00135, 3.5, True)
_add_isotope('Ca', 44, 43.95548, 0.02086, 0.0, True)
_add_isotope('Ca', 46, 45.95369, 0.00004, 0.0, True)
_add_isotope('Ca', 48, 47.95253, 0.00187, 0.0, True)

# =============================================================================
# Scandium
# =============================================================================
_add_isotope('Sc', 45, 44.95591, 1.0, 3.5, True)

# =============================================================================
# Titanium
# =============================================================================
_add_isotope('Ti', 46, 45.95263, 0.0825, 0.0, True)
_add_isotope('Ti', 47, 46.95176, 0.0744, 2.5, True)
_add_isotope('Ti', 48, 47.94795, 0.7372, 0.0, True)
_add_isotope('Ti', 49, 48.94787, 0.0541, 3.5, True)
_add_isotope('Ti', 50, 49.94479, 0.0518, 0.0, True)

# =============================================================================
# Vanadium
# =============================================================================
_add_isotope('V', 50, 49.94716, 0.0025, 6.0, True)
_add_isotope('V', 51, 50.94396, 0.9975, 3.5, True)

# =============================================================================
# Chromium
# =============================================================================
_add_isotope('Cr', 50, 49.94605, 0.0435, 0.0, True)
_add_isotope('Cr', 52, 51.94051, 0.8379, 0.0, True)
_add_isotope('Cr', 53, 52.94065, 0.0950, 1.5, True)
_add_isotope('Cr', 54, 53.93888, 0.0236, 0.0, True)

# =============================================================================
# Manganese
# =============================================================================
_add_isotope('Mn', 55, 54.93805, 1.0, 2.5, True)

# =============================================================================
# Iron - Iron-based superconductors
# =============================================================================
_add_isotope('Fe', 54, 53.93961, 0.0584, 0.0, True)
_add_isotope('Fe', 56, 55.93494, 0.9175, 0.0, True)
_add_isotope('Fe', 57, 56.93540, 0.0212, 0.5, True)
_add_isotope('Fe', 58, 57.93328, 0.0028, 0.0, True)

# =============================================================================
# Cobalt
# =============================================================================
_add_isotope('Co', 59, 58.93320, 1.0, 3.5, True)

# =============================================================================
# Nickel - Nickelate superconductors
# =============================================================================
_add_isotope('Ni', 58, 57.93535, 0.6808, 0.0, True)
_add_isotope('Ni', 60, 59.93079, 0.2622, 0.0, True)
_add_isotope('Ni', 61, 60.93106, 0.0114, 1.5, True)
_add_isotope('Ni', 62, 61.92835, 0.0363, 0.0, True)
_add_isotope('Ni', 64, 63.92797, 0.0093, 0.0, True)

# =============================================================================
# Copper - Critical for cuprate superconductors
# =============================================================================
_add_isotope('Cu', 63, 62.92960, 0.6917, 1.5, True)
_add_isotope('Cu', 65, 64.92779, 0.3083, 1.5, True)

# =============================================================================
# Zinc
# =============================================================================
_add_isotope('Zn', 64, 63.92915, 0.4863, 0.0, True)
_add_isotope('Zn', 66, 65.92603, 0.2790, 0.0, True)
_add_isotope('Zn', 67, 66.92713, 0.0410, 2.5, True)
_add_isotope('Zn', 68, 67.92485, 0.1875, 0.0, True)
_add_isotope('Zn', 70, 69.92532, 0.0062, 0.0, True)

# =============================================================================
# Gallium
# =============================================================================
_add_isotope('Ga', 69, 68.92558, 0.6011, 1.5, True)
_add_isotope('Ga', 71, 70.92470, 0.3989, 1.5, True)

# =============================================================================
# Germanium
# =============================================================================
_add_isotope('Ge', 70, 69.92425, 0.2084, 0.0, True)
_add_isotope('Ge', 72, 71.92208, 0.2754, 0.0, True)
_add_isotope('Ge', 73, 72.92346, 0.0773, 4.5, True)
_add_isotope('Ge', 74, 73.92118, 0.3628, 0.0, True)
_add_isotope('Ge', 76, 75.92140, 0.0761, 0.0, True)

# =============================================================================
# Arsenic - Iron-based superconductors (FeAs planes)
# =============================================================================
_add_isotope('As', 75, 74.92160, 1.0, 1.5, True)

# =============================================================================
# Selenium - FeSe superconductor
# =============================================================================
_add_isotope('Se', 74, 73.92248, 0.0089, 0.0, True)
_add_isotope('Se', 76, 75.91921, 0.0937, 0.0, True)
_add_isotope('Se', 77, 76.91991, 0.0763, 0.5, True)
_add_isotope('Se', 78, 77.91731, 0.2377, 0.0, True)
_add_isotope('Se', 80, 79.91652, 0.4961, 0.0, True)
_add_isotope('Se', 82, 81.91670, 0.0873, 0.0, True)

# =============================================================================
# Bromine
# =============================================================================
_add_isotope('Br', 79, 78.91834, 0.5069, 1.5, True)
_add_isotope('Br', 81, 80.91629, 0.4931, 1.5, True)

# =============================================================================
# Krypton
# =============================================================================
_add_isotope('Kr', 78, 77.92040, 0.0035, 0.0, True)
_add_isotope('Kr', 80, 79.91638, 0.0228, 0.0, True)
_add_isotope('Kr', 82, 81.91348, 0.1158, 0.0, True)
_add_isotope('Kr', 83, 82.91414, 0.1149, 4.5, True)
_add_isotope('Kr', 84, 83.91151, 0.5700, 0.0, True)
_add_isotope('Kr', 86, 85.91062, 0.1730, 0.0, True)

# =============================================================================
# Rubidium - Rb3C60 superconductor
# =============================================================================
_add_isotope('Rb', 85, 84.91179, 0.7217, 2.5, True)
_add_isotope('Rb', 87, 86.90918, 0.2783, 1.5, False, 1.5e18)  # Very long-lived

# =============================================================================
# Strontium - Cuprates (BSCCO)
# =============================================================================
_add_isotope('Sr', 84, 83.91343, 0.0056, 0.0, True)
_add_isotope('Sr', 86, 85.90926, 0.0986, 0.0, True)
_add_isotope('Sr', 87, 86.90888, 0.0700, 4.5, True)
_add_isotope('Sr', 88, 87.90561, 0.8258, 0.0, True)

# =============================================================================
# Yttrium - YBCO superconductor
# =============================================================================
_add_isotope('Y', 89, 88.90585, 1.0, 0.5, True)

# =============================================================================
# Zirconium
# =============================================================================
_add_isotope('Zr', 90, 89.90470, 0.5145, 0.0, True)
_add_isotope('Zr', 91, 90.90565, 0.1122, 2.5, True)
_add_isotope('Zr', 92, 91.90504, 0.1715, 0.0, True)
_add_isotope('Zr', 94, 93.90632, 0.1738, 0.0, True)
_add_isotope('Zr', 96, 95.90828, 0.0280, 0.0, True)

# =============================================================================
# Niobium - A15 superconductors (Nb3Sn, NbTi)
# =============================================================================
_add_isotope('Nb', 93, 92.90638, 1.0, 4.5, True)

# =============================================================================
# Molybdenum
# =============================================================================
_add_isotope('Mo', 92, 91.90681, 0.1484, 0.0, True)
_add_isotope('Mo', 94, 93.90509, 0.0925, 0.0, True)
_add_isotope('Mo', 95, 94.90584, 0.1592, 2.5, True)
_add_isotope('Mo', 96, 95.90468, 0.1668, 0.0, True)
_add_isotope('Mo', 97, 96.90602, 0.0955, 2.5, True)
_add_isotope('Mo', 98, 97.90541, 0.2413, 0.0, True)
_add_isotope('Mo', 100, 99.90748, 0.0963, 0.0, True)

# =============================================================================
# Technetium (radioactive, but included for completeness)
# =============================================================================
_add_isotope('Tc', 98, 97.90722, 0.0, 6.0, False, 1.3e14)
_add_isotope('Tc', 99, 98.90625, 0.0, 4.5, False, 6.7e12)

# =============================================================================
# Ruthenium
# =============================================================================
_add_isotope('Ru', 96, 95.90760, 0.0554, 0.0, True)
_add_isotope('Ru', 98, 97.90529, 0.0187, 0.0, True)
_add_isotope('Ru', 99, 98.90594, 0.1276, 2.5, True)
_add_isotope('Ru', 100, 99.90422, 0.1260, 0.0, True)
_add_isotope('Ru', 101, 100.90558, 0.1706, 2.5, True)
_add_isotope('Ru', 102, 101.90435, 0.3155, 0.0, True)
_add_isotope('Ru', 104, 103.90543, 0.1862, 0.0, True)

# =============================================================================
# Rhodium
# =============================================================================
_add_isotope('Rh', 103, 102.90550, 1.0, 0.5, True)

# =============================================================================
# Palladium
# =============================================================================
_add_isotope('Pd', 102, 101.90561, 0.0102, 0.0, True)
_add_isotope('Pd', 104, 103.90404, 0.1114, 0.0, True)
_add_isotope('Pd', 105, 104.90508, 0.2233, 2.5, True)
_add_isotope('Pd', 106, 105.90348, 0.2733, 0.0, True)
_add_isotope('Pd', 108, 107.90389, 0.2646, 0.0, True)
_add_isotope('Pd', 110, 109.90517, 0.1172, 0.0, True)

# =============================================================================
# Silver
# =============================================================================
_add_isotope('Ag', 107, 106.90509, 0.5184, 0.5, True)
_add_isotope('Ag', 109, 108.90476, 0.4816, 0.5, True)

# =============================================================================
# Cadmium
# =============================================================================
_add_isotope('Cd', 106, 105.90646, 0.0125, 0.0, True)
_add_isotope('Cd', 108, 107.90418, 0.0089, 0.0, True)
_add_isotope('Cd', 110, 109.90300, 0.1249, 0.0, True)
_add_isotope('Cd', 111, 110.90418, 0.1280, 0.5, True)
_add_isotope('Cd', 112, 111.90276, 0.2413, 0.0, True)
_add_isotope('Cd', 113, 112.90440, 0.1222, 0.5, True)
_add_isotope('Cd', 114, 113.90336, 0.2873, 0.0, True)
_add_isotope('Cd', 116, 115.90476, 0.0749, 0.0, True)

# =============================================================================
# Indium
# =============================================================================
_add_isotope('In', 113, 112.90406, 0.0429, 4.5, True)
_add_isotope('In', 115, 114.90388, 0.9571, 4.5, False, 1.4e22)  # Very long-lived

# =============================================================================
# Tin - A15 superconductors (Nb3Sn)
# =============================================================================
_add_isotope('Sn', 112, 111.90482, 0.0097, 0.0, True)
_add_isotope('Sn', 114, 113.90278, 0.0066, 0.0, True)
_add_isotope('Sn', 115, 114.90334, 0.0034, 0.5, True)
_add_isotope('Sn', 116, 115.90174, 0.1454, 0.0, True)
_add_isotope('Sn', 117, 116.90295, 0.0768, 0.5, True)
_add_isotope('Sn', 118, 117.90160, 0.2422, 0.0, True)
_add_isotope('Sn', 119, 118.90331, 0.0859, 0.5, True)
_add_isotope('Sn', 120, 119.90220, 0.3258, 0.0, True)
_add_isotope('Sn', 122, 121.90344, 0.0463, 0.0, True)
_add_isotope('Sn', 124, 123.90527, 0.0579, 0.0, True)

# =============================================================================
# Antimony
# =============================================================================
_add_isotope('Sb', 121, 120.90382, 0.5721, 2.5, True)
_add_isotope('Sb', 123, 122.90421, 0.4279, 3.5, True)

# =============================================================================
# Tellurium - FeTe superconductor
# =============================================================================
_add_isotope('Te', 120, 119.90402, 0.0009, 0.0, True)
_add_isotope('Te', 122, 121.90304, 0.0255, 0.0, True)
_add_isotope('Te', 123, 122.90427, 0.0089, 0.5, True)
_add_isotope('Te', 124, 123.90282, 0.0474, 0.0, True)
_add_isotope('Te', 125, 124.90443, 0.0707, 0.5, True)
_add_isotope('Te', 126, 125.90331, 0.1884, 0.0, True)
_add_isotope('Te', 128, 127.90446, 0.3174, 0.0, True)
_add_isotope('Te', 130, 129.90622, 0.3408, 0.0, True)

# =============================================================================
# Iodine
# =============================================================================
_add_isotope('I', 127, 126.90447, 1.0, 2.5, True)

# =============================================================================
# Xenon
# =============================================================================
_add_isotope('Xe', 124, 123.90589, 0.0009, 0.0, True)
_add_isotope('Xe', 126, 125.90430, 0.0009, 0.0, True)
_add_isotope('Xe', 128, 127.90353, 0.0192, 0.0, True)
_add_isotope('Xe', 129, 128.90478, 0.2644, 0.5, True)
_add_isotope('Xe', 130, 129.90351, 0.0408, 0.0, True)
_add_isotope('Xe', 131, 130.90508, 0.2118, 1.5, True)
_add_isotope('Xe', 132, 131.90415, 0.2689, 0.0, True)
_add_isotope('Xe', 134, 133.90539, 0.1044, 0.0, True)
_add_isotope('Xe', 136, 135.90722, 0.0887, 0.0, True)

# =============================================================================
# Cesium - Cs3C60 superconductor
# =============================================================================
_add_isotope('Cs', 133, 132.90545, 1.0, 3.5, True)

# =============================================================================
# Barium - Cuprates (YBCO, BSCCO)
# =============================================================================
_add_isotope('Ba', 130, 129.90632, 0.0011, 0.0, True)
_add_isotope('Ba', 132, 131.90506, 0.0010, 0.0, True)
_add_isotope('Ba', 134, 133.90450, 0.0242, 0.0, True)
_add_isotope('Ba', 135, 134.90569, 0.0659, 1.5, True)
_add_isotope('Ba', 136, 135.90457, 0.0785, 0.0, True)
_add_isotope('Ba', 137, 136.90582, 0.1123, 1.5, True)
_add_isotope('Ba', 138, 137.90524, 0.7170, 0.0, True)

# =============================================================================
# Lanthanum - Cuprates, LaH10 hydride
# =============================================================================
_add_isotope('La', 138, 137.90711, 0.0009, 5.0, True)
_add_isotope('La', 139, 138.90636, 0.9991, 3.5, True)

# =============================================================================
# Cerium
# =============================================================================
_add_isotope('Ce', 136, 135.90714, 0.0019, 0.0, True)
_add_isotope('Ce', 138, 137.90599, 0.0025, 0.0, True)
_add_isotope('Ce', 140, 139.90544, 0.8848, 0.0, True)
_add_isotope('Ce', 142, 141.90924, 0.1108, 0.0, True)

# =============================================================================
# Praseodymium
# =============================================================================
_add_isotope('Pr', 141, 140.90765, 1.0, 2.5, True)

# =============================================================================
# Neodymium
# =============================================================================
_add_isotope('Nd', 142, 141.90773, 0.2713, 0.0, True)
_add_isotope('Nd', 143, 142.90981, 0.1218, 3.5, True)
_add_isotope('Nd', 144, 143.91009, 0.2380, 0.0, True)
_add_isotope('Nd', 145, 144.91257, 0.0830, 3.5, True)
_add_isotope('Nd', 146, 145.91312, 0.1719, 0.0, True)
_add_isotope('Nd', 148, 147.91689, 0.0576, 0.0, True)
_add_isotope('Nd', 150, 149.92090, 0.0564, 0.0, True)

# =============================================================================
# Continue with remaining lanthanides and actinides...
# (Adding key elements for superconductivity)
# =============================================================================

# Samarium
_add_isotope('Sm', 144, 143.91199, 0.0307, 0.0, True)
_add_isotope('Sm', 147, 146.91490, 0.1499, 3.5, False, 3.4e18)
_add_isotope('Sm', 148, 147.91483, 0.1124, 0.0, True)
_add_isotope('Sm', 149, 148.91719, 0.1382, 3.5, True)
_add_isotope('Sm', 150, 149.91728, 0.0738, 0.0, True)
_add_isotope('Sm', 152, 151.91973, 0.2675, 0.0, True)
_add_isotope('Sm', 154, 153.92222, 0.2275, 0.0, True)

# Europium
_add_isotope('Eu', 151, 150.91986, 0.4781, 2.5, True)
_add_isotope('Eu', 153, 152.92123, 0.5219, 2.5, True)

# Gadolinium
_add_isotope('Gd', 152, 151.91979, 0.0020, 0.0, True)
_add_isotope('Gd', 154, 153.92087, 0.0218, 0.0, True)
_add_isotope('Gd', 155, 154.92263, 0.1480, 1.5, True)
_add_isotope('Gd', 156, 155.92213, 0.2047, 0.0, True)
_add_isotope('Gd', 157, 156.92396, 0.1565, 1.5, True)
_add_isotope('Gd', 158, 157.92411, 0.2484, 0.0, True)
_add_isotope('Gd', 160, 159.92710, 0.2186, 0.0, True)

# Terbium
_add_isotope('Tb', 159, 158.92535, 1.0, 1.5, True)

# Dysprosium
_add_isotope('Dy', 156, 155.92428, 0.0006, 0.0, True)
_add_isotope('Dy', 158, 157.92442, 0.0010, 0.0, True)
_add_isotope('Dy', 160, 159.92520, 0.0234, 0.0, True)
_add_isotope('Dy', 161, 160.92694, 0.1891, 2.5, True)
_add_isotope('Dy', 162, 161.92680, 0.2551, 0.0, True)
_add_isotope('Dy', 163, 162.92874, 0.2490, 2.5, True)
_add_isotope('Dy', 164, 163.92918, 0.2818, 0.0, True)

# Holmium
_add_isotope('Ho', 165, 164.93033, 1.0, 3.5, True)

# Erbium
_add_isotope('Er', 162, 161.92879, 0.0014, 0.0, True)
_add_isotope('Er', 164, 163.92921, 0.0161, 0.0, True)
_add_isotope('Er', 166, 165.93030, 0.3361, 0.0, True)
_add_isotope('Er', 167, 166.93205, 0.2293, 3.5, True)
_add_isotope('Er', 168, 167.93238, 0.2678, 0.0, True)
_add_isotope('Er', 170, 169.93547, 0.1493, 0.0, True)

# Thulium
_add_isotope('Tm', 169, 168.93422, 1.0, 0.5, True)

# Ytterbium
_add_isotope('Yb', 168, 167.93390, 0.0013, 0.0, True)
_add_isotope('Yb', 170, 169.93477, 0.0304, 0.0, True)
_add_isotope('Yb', 171, 170.93633, 0.1428, 0.5, True)
_add_isotope('Yb', 172, 171.93639, 0.2183, 0.0, True)
_add_isotope('Yb', 173, 172.93822, 0.1613, 2.5, True)
_add_isotope('Yb', 174, 173.93887, 0.3183, 0.0, True)
_add_isotope('Yb', 176, 175.94258, 0.1276, 0.0, True)

# Lutetium
_add_isotope('Lu', 175, 174.94078, 0.9741, 3.5, True)
_add_isotope('Lu', 176, 175.94269, 0.0259, 7.0, False, 1.2e18)

# =============================================================================
# Heavy elements (Hf - U) - Important for exotic superconductors
# =============================================================================

# Hafnium
_add_isotope('Hf', 174, 173.94004, 0.0016, 0.0, True)
_add_isotope('Hf', 176, 175.94141, 0.0526, 0.0, True)
_add_isotope('Hf', 177, 176.94323, 0.1860, 3.5, True)
_add_isotope('Hf', 178, 177.94370, 0.2728, 0.0, True)
_add_isotope('Hf', 179, 178.94582, 0.1362, 4.5, True)
_add_isotope('Hf', 180, 179.94655, 0.3508, 0.0, True)

# Tantalum
_add_isotope('Ta', 180, 179.94747, 0.0001, 8.0, True)
_add_isotope('Ta', 181, 180.94800, 0.9999, 3.5, True)

# Tungsten
_add_isotope('W', 180, 179.94671, 0.0012, 0.0, True)
_add_isotope('W', 182, 181.94820, 0.2650, 0.0, True)
_add_isotope('W', 183, 182.95022, 0.1431, 0.5, True)
_add_isotope('W', 184, 183.95093, 0.3064, 0.0, True)
_add_isotope('W', 186, 185.95436, 0.2843, 0.0, True)

# Rhenium
_add_isotope('Re', 185, 184.95296, 0.3740, 2.5, True)
_add_isotope('Re', 187, 186.95575, 0.6260, 2.5, False, 1.4e18)

# Osmium
_add_isotope('Os', 184, 183.95249, 0.0002, 0.0, True)
_add_isotope('Os', 186, 185.95384, 0.0159, 0.0, True)
_add_isotope('Os', 187, 186.95575, 0.0196, 0.5, True)
_add_isotope('Os', 188, 187.95584, 0.1324, 0.0, True)
_add_isotope('Os', 189, 188.95814, 0.1615, 1.5, True)
_add_isotope('Os', 190, 189.95844, 0.2626, 0.0, True)
_add_isotope('Os', 192, 191.96148, 0.4078, 0.0, True)

# Iridium
_add_isotope('Ir', 191, 190.96059, 0.373, 1.5, True)
_add_isotope('Ir', 193, 192.96292, 0.627, 1.5, True)

# Platinum
_add_isotope('Pt', 190, 189.95993, 0.0001, 0.0, True)
_add_isotope('Pt', 192, 191.96104, 0.0079, 0.0, True)
_add_isotope('Pt', 194, 193.96268, 0.3290, 0.0, True)
_add_isotope('Pt', 195, 194.96479, 0.3380, 0.5, True)
_add_isotope('Pt', 196, 195.96495, 0.2530, 0.0, True)
_add_isotope('Pt', 198, 197.96789, 0.0720, 0.0, True)

# Gold
_add_isotope('Au', 197, 196.96657, 1.0, 1.5, True)

# Mercury - Conventional superconductor (first discovered)
_add_isotope('Hg', 196, 195.96583, 0.0015, 0.0, True)
_add_isotope('Hg', 198, 197.96676, 0.0997, 0.0, True)
_add_isotope('Hg', 199, 198.96828, 0.1687, 0.5, True)
_add_isotope('Hg', 200, 199.96832, 0.2310, 0.0, True)
_add_isotope('Hg', 201, 200.97030, 0.1318, 1.5, True)
_add_isotope('Hg', 202, 201.97064, 0.2986, 0.0, True)
_add_isotope('Hg', 204, 203.97349, 0.0687, 0.0, True)

# Thallium - Cuprates (Tl-based)
_add_isotope('Tl', 203, 202.97234, 0.2952, 0.5, True)
_add_isotope('Tl', 205, 204.97441, 0.7048, 0.5, True)

# Lead - Conventional superconductor
_add_isotope('Pb', 204, 203.97304, 0.014, 0.0, True)
_add_isotope('Pb', 206, 205.97446, 0.241, 0.0, True)
_add_isotope('Pb', 207, 206.97590, 0.221, 0.5, True)
_add_isotope('Pb', 208, 207.97665, 0.524, 0.0, True)

# Bismuth - Cuprates (BSCCO)
_add_isotope('Bi', 209, 208.98040, 1.0, 4.5, False, 6.0e26)  # Effectively stable

# Thorium
_add_isotope('Th', 232, 232.03805, 1.0, 0.0, False, 4.4e17)

# Uranium
_add_isotope('U', 234, 234.04095, 0.00005, 0.0, False, 7.7e12)
_add_isotope('U', 235, 235.04393, 0.00720, 3.5, False, 2.2e16)
_add_isotope('U', 238, 238.05079, 0.99275, 0.0, False, 1.4e17)


# =============================================================================
# Utility functions
# =============================================================================

def get_isotope(symbol: str, mass_number: int) -> Optional[IsotopeData]:
    """Get isotope data by symbol and mass number."""
    if symbol in ISOTOPE_DATABASE:
        return ISOTOPE_DATABASE[symbol].get(mass_number)
    return None


def get_all_isotopes(symbol: str) -> List[IsotopeData]:
    """Get all isotopes for an element."""
    if symbol in ISOTOPE_DATABASE:
        return list(ISOTOPE_DATABASE[symbol].values())
    return []


def get_stable_isotopes(symbol: str) -> List[IsotopeData]:
    """Get only stable isotopes for an element."""
    return [iso for iso in get_all_isotopes(symbol) if iso.is_stable]


def get_most_abundant_isotope(symbol: str) -> Optional[IsotopeData]:
    """Get the most naturally abundant isotope for an element."""
    isotopes = get_all_isotopes(symbol)
    if not isotopes:
        return None
    return max(isotopes, key=lambda x: x.natural_abundance)


def get_average_atomic_mass(symbol: str) -> float:
    """Get natural abundance weighted average atomic mass."""
    isotopes = get_all_isotopes(symbol)
    if not isotopes:
        return 0.0

    total_mass = sum(iso.atomic_mass * iso.natural_abundance for iso in isotopes)
    total_abundance = sum(iso.natural_abundance for iso in isotopes)

    if total_abundance == 0:
        # All synthetic - return mass of most stable
        return isotopes[0].atomic_mass

    return total_mass / total_abundance


def parse_isotope_notation(notation: str) -> Tuple[str, int]:
    """
    Parse isotope notation like '2H', '18O', 'D', 'T'.

    Returns:
        Tuple of (element_symbol, mass_number)
    """
    # Special cases
    if notation == 'D':
        return ('H', 2)
    if notation == 'T':
        return ('H', 3)

    # Standard notation: mass_number + symbol (e.g., '18O', '13C')
    import re
    match = re.match(r'^(\d+)([A-Z][a-z]?)$', notation)
    if match:
        return (match.group(2), int(match.group(1)))

    # Just element symbol - use most abundant
    if notation in ISOTOPE_DATABASE:
        most_abundant = get_most_abundant_isotope(notation)
        if most_abundant:
            return (notation, most_abundant.mass_number)

    raise ValueError(f"Cannot parse isotope notation: {notation}")


def get_isotope_mass_ratio(symbol: str, mass_number_1: int, mass_number_2: int) -> float:
    """
    Get mass ratio between two isotopes (for isotope effect calculations).

    The isotope effect in BCS superconductors follows:
        Tc ∝ M^(-α) where α ≈ 0.5

    So Tc_2 / Tc_1 ≈ (M_1 / M_2)^0.5
    """
    iso1 = get_isotope(symbol, mass_number_1)
    iso2 = get_isotope(symbol, mass_number_2)

    if iso1 is None or iso2 is None:
        raise ValueError(f"Isotope not found: {symbol}-{mass_number_1} or {symbol}-{mass_number_2}")

    return iso1.atomic_mass / iso2.atomic_mass


def estimate_isotope_effect(symbol: str, mass_number_1: int, mass_number_2: int,
                            tc_1: float, alpha: float = 0.5) -> float:
    """
    Estimate Tc change from isotope substitution using BCS isotope effect.

    Args:
        symbol: Element symbol
        mass_number_1: Original isotope mass number
        mass_number_2: New isotope mass number
        tc_1: Original Tc (K)
        alpha: Isotope effect exponent (0.5 for BCS, varies for unconventional)

    Returns:
        Estimated Tc with new isotope
    """
    mass_ratio = get_isotope_mass_ratio(symbol, mass_number_1, mass_number_2)
    return tc_1 * (mass_ratio ** alpha)


# Count total isotopes in database
TOTAL_ISOTOPES = sum(len(isotopes) for isotopes in ISOTOPE_DATABASE.values())
TOTAL_ELEMENTS = len(ISOTOPE_DATABASE)

print(f"Isotope database loaded: {TOTAL_ISOTOPES} isotopes for {TOTAL_ELEMENTS} elements")
