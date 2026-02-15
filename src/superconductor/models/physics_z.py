"""
Physics-supervised Z coordinate map for the 2048-dim latent vector (V12.31).

Defines named coordinate blocks encoding specific physical quantities.
All coordinates are indices into z[batch, 2048].

Blocks 1-11 (coords 0-511) are supervised/constrained.
Block 12 (coords 512-2047) is unsupervised discovery space.

No architectural changes to FullMaterialsVAE — physics meaning is enforced
purely via loss functions. Existing checkpoints load unchanged.
"""


class PhysicsZ:
    """Defines the coordinate map for the 2048-dim Z vector.

    All coordinates are indices into z[batch, 2048].
    Blocks 1-11 (coords 0-511) are supervised/constrained.
    Block 12 (coords 512-2047) is unsupervised discovery space.
    """

    # === BLOCK 1: Ginzburg-Landau (0-19) ===
    KAPPA = 0          # GL parameter kappa = lambda/xi (dimensionless)
    XI = 1             # Coherence length xi (normalized)
    LAMBDA_L = 2       # London penetration depth lambda_L (normalized)
    DELTA0 = 3         # Zero-temp energy gap Delta_0 (normalized)
    HC = 4             # Thermodynamic critical field H_c (normalized)
    HC1 = 5            # Lower critical field H_c1 (normalized)
    HC2 = 6            # Upper critical field H_c2 (normalized)
    ALPHA_GL = 7       # GL alpha coefficient (normalized)
    BETA_GL = 8        # GL beta coefficient (normalized)
    E_COND = 9         # Condensation energy density (normalized)
    SIGMA_NS = 10      # N-S surface energy (normalized)
    N_S = 11           # Superfluid density (normalized)
    DELTA_KAPPA = 12   # Maki parameter (dimensionless)
    ETA = 13           # Flux flow viscosity (normalized)
    # 14-19: reserved for anisotropic GL extensions
    GL_START, GL_END = 0, 20

    # === BLOCK 2: BCS/Microscopic (20-49) ===
    V_F = 20           # Fermi velocity
    K_F = 21           # Fermi wavevector
    E_F = 22           # Fermi energy
    N_EF = 23          # DOS at Fermi level
    LAMBDA_EP = 24     # Electron-phonon coupling
    MU_STAR = 25       # Coulomb pseudopotential
    OMEGA_D = 26       # Debye frequency
    THETA_D = 27       # Debye temperature
    OMEGA_LOG = 28     # Log-average phonon frequency
    GAP_RATIO = 29     # 2*Delta_0 / k_B*Tc (BCS: 3.528)
    HEAT_JUMP = 30     # Delta_C / gamma*Tc (BCS: 1.43)
    GAMMA_N = 31       # Normal Sommerfeld coefficient
    GAMMA_S = 32       # Residual Sommerfeld in SC state
    M_STAR = 33        # Effective mass
    RHO_N = 34         # Normal state resistivity
    L_MFP = 35         # Mean free path
    RRR = 36           # Residual resistivity ratio
    GAP_DEBYE = 37     # 2*Delta_0 / k_B*Theta_D
    TC_OMEGA = 38      # Tc / omega_log (reduced Tc)
    # 39-49: reserved
    BCS_START, BCS_END = 20, 50

    # === BLOCK 3: Eliashberg (50-69) ===
    # Spectral function parameters and strong-coupling corrections
    ALPHA2F_PEAK = 50     # Peak of alpha^2*F(omega)
    ALPHA2F_WIDTH = 51    # Width of spectral function
    ALPHA2F_CENTROID = 52 # Centroid frequency
    LAMBDA_TR = 53        # Transport coupling constant
    LAMBDA_OPT = 54       # Optical coupling constant
    MU_STAR_ELIASHBERG = 55  # Eliashberg mu* (may differ from BCS)
    STRONG_COUPLING_CORR = 56  # Strong coupling correction factor
    RETARDATION = 57      # Retardation parameter omega_log / E_F
    # 58-69: reserved
    ELIASHBERG_START, ELIASHBERG_END = 50, 70

    # === BLOCK 4: Unconventional (70-109) ===
    GAP_SYM = 70       # Gap symmetry class (encoded)
    GAP_NODE_TYPE = 71  # Node topology (point, line, etc.)
    GAP_ANISO = 72     # Gap anisotropy ratio
    D_WAVE_AMP = 73    # d-wave amplitude
    S_WAVE_AMP = 74    # s-wave amplitude
    P_WAVE_AMP = 75    # p-wave amplitude
    SPIN_SINGLET = 76  # Spin-singlet fraction
    SPIN_TRIPLET = 77  # Spin-triplet fraction
    NEMATIC_ORDER = 78 # Nematic order parameter
    DOPING = 79        # Doping level p
    OPTIMAL_DOPING = 80  # Optimal doping for this family
    CU_O_APICAL = 81  # Apical Cu-O distance (cuprates)
    CU_O_PLANAR = 82  # Planar Cu-O distance (cuprates)
    N_CUO2_LAYERS = 83  # Number of CuO2 layers
    CHARGE_TRANSFER_GAP = 84  # Charge transfer energy
    SUPEREXCHANGE_J = 85  # Superexchange coupling
    AF_CORR_LENGTH = 86  # Antiferromagnetic correlation length
    # 87-109: reserved
    UNCONVENTIONAL_START, UNCONVENTIONAL_END = 70, 110

    # === BLOCK 5: Structural (110-159) ===
    SG_NUM = 110       # Space group number
    SG_ENCODED = 111   # Space group (learned encoding)
    CRYSTAL_SYS = 112  # Crystal system (1-7)
    LATTICE_A = 113    # Lattice parameter a
    LATTICE_B = 114    # Lattice parameter b
    LATTICE_C = 115    # Lattice parameter c
    LATTICE_ALPHA = 116  # Lattice angle alpha
    LATTICE_BETA = 117   # Lattice angle beta
    LATTICE_GAMMA = 118  # Lattice angle gamma
    VOLUME = 119       # Unit cell volume
    Z_FORMULA = 120    # Formula units per cell
    PACKING_FRAC = 121 # Packing fraction
    BOND_LENGTH_AVG = 122  # Average bond length
    BOND_LENGTH_CRIT = 123 # Critical bond length (e.g., Cu-O)
    COORD_NUM_AVG = 124  # Average coordination number
    # 125-159: reserved
    STRUCTURAL_START, STRUCTURAL_END = 110, 160

    # === BLOCK 6: Electronic (160-209) ===
    N_VALENCE = 160    # Valence electron count
    BAND_WIDTH = 161   # Total bandwidth
    BAND_GAP = 162     # Band gap (0 for metals)
    DOS_SHAPE = 163    # DOS shape parameter (van Hove singularity proximity)
    PLASMA_FREQ = 164  # Plasma frequency
    DRUDE_WEIGHT = 165 # Drude weight
    HALL_COEFF = 166   # Hall coefficient
    SEEBECK = 167      # Seebeck coefficient
    ORBITAL_CHAR = 168 # Dominant orbital character
    SPIN_ORBIT = 169   # Spin-orbit coupling strength
    # 170-209: reserved
    ELECTRONIC_START, ELECTRONIC_END = 160, 210

    # === BLOCK 7: Thermodynamic (210-269) ===
    TC = 210           # Critical temperature (primary)
    TC_ONSET = 211     # Onset Tc
    TC_MIDPOINT = 212  # Midpoint Tc
    TC_ZERO = 213      # Zero-resistance Tc
    DELTA_TC = 214     # Transition width
    HC_0 = 215         # Zero-temperature critical field
    SLOPE_HC = 216     # dHc2/dT at Tc
    JUMP_CP = 217      # Specific heat jump at Tc
    GAMMA_NORMAL = 218 # Normal-state specific heat coefficient
    BETA_PHONON = 219  # Phonon specific heat coefficient
    ENTROPY_SC = 220   # Superconducting entropy
    # 221-269: reserved
    THERMO_START, THERMO_END = 210, 270

    # === BLOCK 8: Compositional (270-339) ===
    N_ELEMENTS = 270   # Number of distinct elements
    MW = 271           # Molecular weight
    X_H = 272          # Hydrogen fraction
    Z_AVG = 273        # Average atomic number
    Z_MAX = 274        # Maximum atomic number
    EN_AVG = 275       # Average electronegativity
    EN_DIFF = 276      # Electronegativity difference
    R_AVG = 277        # Average atomic radius
    R_RATIO = 278      # Radius ratio max/min
    VEC = 279          # Valence electron concentration
    EA_RATIO = 280     # e/a ratio
    DELTA_SIZE = 281   # Size mismatch parameter
    # 282-286: reserved
    D_ORBITAL_FRAC = 287  # d-electron element fraction
    F_ORBITAL_FRAC = 288  # f-electron element fraction
    IE_AVG = 289       # Average ionization energy
    TM_AVG = 285       # Average melting temperature (uses reserved coord)
    # 291-339: reserved
    COMP_START, COMP_END = 270, 340

    # === BLOCK 9: Cobordism (340-399) ===
    E_VORTEX = 340     # Vortex nucleation energy (pi_1 cost)
    E_DOMAIN = 341     # Domain wall energy (pi_0 cost)
    E_MONOPOLE = 342   # Monopole energy (pi_2 cost)
    E_DEFECT_MIN = 343 # Min defect energy (predicted to correlate with Tc)
    TYPE_I_II = 344    # kappa - 1/sqrt(2) (sign = type)
    # 345-399: reserved
    COBORDISM_START, COBORDISM_END = 340, 400

    # === BLOCK 10: Dimensionless Ratios (400-449) ===
    TC_THETA_D = 400   # Tc / Theta_D
    TC_EF = 401        # k_B*Tc / E_F
    LAMBDA_OVER_XI = 402  # lambda / xi (redundant with kappa, cross-check)
    XI_L = 403         # xi_0 / l_mfp (dirty/clean)
    XI_A = 404         # xi_0 / a (coherence/lattice)
    # 405-449: reserved
    RATIOS_START, RATIOS_END = 400, 450

    # === BLOCK 11: Magpie Encoding (450-511) ===
    MAGPIE_START, MAGPIE_END = 450, 512

    # === BLOCK 12: Discovery (512-2047) ===
    DISCOVERY_START, DISCOVERY_END = 512, 2048

    # Total supervised coordinates
    N_SUPERVISED = 512
    N_DISCOVERY = 1536
    N_TOTAL = 2048

    @classmethod
    def get_block_ranges(cls):
        """Return dict of block_name -> (start, end) for all blocks."""
        return {
            'gl': (cls.GL_START, cls.GL_END),
            'bcs': (cls.BCS_START, cls.BCS_END),
            'eliashberg': (cls.ELIASHBERG_START, cls.ELIASHBERG_END),
            'unconventional': (cls.UNCONVENTIONAL_START, cls.UNCONVENTIONAL_END),
            'structural': (cls.STRUCTURAL_START, cls.STRUCTURAL_END),
            'electronic': (cls.ELECTRONIC_START, cls.ELECTRONIC_END),
            'thermodynamic': (cls.THERMO_START, cls.THERMO_END),
            'compositional': (cls.COMP_START, cls.COMP_END),
            'cobordism': (cls.COBORDISM_START, cls.COBORDISM_END),
            'ratios': (cls.RATIOS_START, cls.RATIOS_END),
            'magpie': (cls.MAGPIE_START, cls.MAGPIE_END),
            'discovery': (cls.DISCOVERY_START, cls.DISCOVERY_END),
        }

    @classmethod
    def get_block_sizes(cls):
        """Return dict of block_name -> size for all blocks."""
        return {name: end - start for name, (start, end) in cls.get_block_ranges().items()}

    @classmethod
    def get_supervised_blocks(cls):
        """Return only supervised blocks (not discovery)."""
        ranges = cls.get_block_ranges()
        return {k: v for k, v in ranges.items() if k != 'discovery'}


# Physical constants (SI)
PHI0 = 2.067833848e-15     # Magnetic flux quantum (Wb)
MU0 = 1.25663706212e-6     # Vacuum permeability (H/m)
KB = 1.380649e-23           # Boltzmann constant (J/K)
HBAR = 1.054571817e-34      # Reduced Planck constant (J*s)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
M_E = 9.1093837015e-31      # Electron mass (kg)
