"""
FractionAwareTokenizer — V14.0 Semantic Fraction + Isotope Tokenization.

Vocabulary structure:
    [0] PAD  [1] BOS  [2] EOS  [3] UNK  [4] FRAC_UNK    (5 special tokens)
    [5..122] H, He, Li, ..., Og                            (118 element tokens)
    [123..142] 1, 2, 3, ..., 20                            (20 integer tokens)
    [143..4354] FRAC:1/2, FRAC:1/4, FRAC:3/4, ...         (4212 fraction tokens from vocab)
    [4355] ISO_UNK: unknown isotope fallback                (1 special isotope token)
    [4356..4646] ISO:1H, ISO:2H, ISO:3H, ...              (291 isotope tokens from vocab)

Key features:
    - Single token per fraction: (17/20) → FRAC:17/20 (was 7 tokens)
    - Single token per isotope: {18}O → ISO:18O (V14.0)
    - Built-in GCD canonicalization during encode
    - Integers 1-20 as direct tokens; integers > 20 → UNK
    - Round-trip encode/decode preserves formula exactly
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Full periodic table (index 0 is empty placeholder, 1-118 are real elements)
ELEMENTS = [
    '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
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

# Set of valid element symbols for fast lookup
ELEMENT_SET = set(ELEMENTS[1:])  # Skip empty string at index 0

# Special token indices
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
FRAC_UNK_IDX = 4

# Special token strings
PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
FRAC_UNK_TOKEN = '<FRAC_UNK>'
ISO_UNK_TOKEN = '<ISO_UNK>'

N_SPECIAL = 5
N_ELEMENTS = 118
MAX_INTEGER = 20

# Regex to tokenize formulas: matches elements, fractions in parens, or bare integers
# Order: fraction (greedy), then element (2-char before 1-char), then integer
_FORMULA_PATTERN = re.compile(
    r'\((\d+)/(\d+)\)'           # Group 1,2: fraction in parentheses
    r'|([A-Z][a-z]?)'           # Group 3: element symbol
    r'|(\d+)'                    # Group 4: bare integer subscript
)

# V14.0: Isotope-aware regex — adds {mass}Element matching (highest priority)
_ISOTOPE_FORMULA_PATTERN = re.compile(
    r'\{(\d+)\}([A-Z][a-z]?)'   # Group 1,2: isotope {mass}Element
    r'|\((\d+)/(\d+)\)'          # Group 3,4: fraction in parentheses
    r'|([A-Z][a-z]?)'            # Group 5: element symbol
    r'|(\d+)'                     # Group 6: bare integer subscript
)


class FractionAwareTokenizer:
    """Semantic fraction + isotope tokenizer for superconductor formula generation.

    Replaces digit-by-digit fraction tokenization with single semantic tokens.
    Each fraction (p/q) becomes one FRAC:p/q token instead of multiple character tokens.
    V14.0: Each isotope {mass}Element becomes one ISO:massSymbol token.

    Args:
        fraction_vocab_path: Path to fraction_vocab.json (built by build_fraction_vocab.py)
        max_len: Maximum sequence length (including BOS/EOS)
        isotope_vocab_path: Path to isotope_vocab.json (built by build_isotope_vocab.py, V14.0)
    """

    def __init__(self, fraction_vocab_path: str = None, max_len: int = 60,
                 isotope_vocab_path: str = None):
        self.max_len = max_len

        # Build base vocabulary: special tokens + elements + integers
        self._build_base_vocab()

        # Load fraction vocabulary if provided
        self._fraction_list = []  # List of "num/den" strings
        self._fraction_to_id = {}  # "num/den" -> token index
        self._fraction_to_value = {}  # token_id -> float value
        self._fraction_vocab_meta = {}

        if fraction_vocab_path is not None:
            self._load_fraction_vocab(fraction_vocab_path)

        # V14.0: Load isotope vocabulary if provided
        self._isotope_list = []  # List of "massSymbol" strings (e.g., "18O", "2H")
        self._isotope_to_id = {}  # "massSymbol" -> token index
        self._iso_offset = 0  # Start index of isotope tokens
        self._iso_unk_idx = None  # ISO_UNK token index
        self._isotope_vocab_meta = {}

        if isotope_vocab_path is not None:
            self._load_isotope_vocab(isotope_vocab_path)

    def _build_base_vocab(self):
        """Build the base vocabulary (special + elements + integers)."""
        self._token_to_id = {}
        self._id_to_token = {}

        # Special tokens [0..4]
        specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, FRAC_UNK_TOKEN]
        for i, tok in enumerate(specials):
            self._token_to_id[tok] = i
            self._id_to_token[i] = tok

        # Element tokens [5..122]
        for i, elem in enumerate(ELEMENTS[1:], start=N_SPECIAL):
            self._token_to_id[elem] = i
            self._id_to_token[i] = elem

        # Integer tokens [123..142]: "1" through "20"
        self._int_offset = N_SPECIAL + N_ELEMENTS  # 123
        for val in range(1, MAX_INTEGER + 1):
            idx = self._int_offset + val - 1  # 1 -> 123, 20 -> 142
            tok = str(val)
            self._token_to_id[tok] = idx
            self._id_to_token[idx] = tok

        self._frac_offset = self._int_offset + MAX_INTEGER  # 143

    def _load_fraction_vocab(self, path: str):
        """Load fraction vocabulary from JSON file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        self._fraction_list = vocab_data['fractions']
        self._fraction_vocab_meta = {
            k: v for k, v in vocab_data.items() if k not in ('fractions', 'fraction_to_id', 'fraction_counts')
        }

        # Build fraction token mappings
        for i, frac_str in enumerate(self._fraction_list):
            token_id = self._frac_offset + i
            frac_token = f"FRAC:{frac_str}"
            self._token_to_id[frac_token] = token_id
            self._id_to_token[token_id] = frac_token

            # Also store the fraction string -> id mapping for encoding
            self._fraction_to_id[frac_str] = token_id

            # Parse float value for fraction_token_to_value()
            parts = frac_str.split('/')
            self._fraction_to_value[token_id] = int(parts[0]) / int(parts[1])

    def _load_isotope_vocab(self, path: str):
        """Load isotope vocabulary from JSON file (V14.0)."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        self._isotope_list = vocab_data['isotopes']
        self._isotope_vocab_meta = {
            k: v for k, v in vocab_data.items() if k not in ('isotopes',)
        }

        # ISO_UNK sits right after all fraction tokens
        frac_end = self._frac_offset + len(self._fraction_list)
        self._iso_unk_idx = frac_end
        self._token_to_id[ISO_UNK_TOKEN] = self._iso_unk_idx
        self._id_to_token[self._iso_unk_idx] = ISO_UNK_TOKEN

        # Isotope tokens start after ISO_UNK
        self._iso_offset = self._iso_unk_idx + 1

        for i, iso_str in enumerate(self._isotope_list):
            token_id = self._iso_offset + i
            iso_token = f"ISO:{iso_str}"
            self._token_to_id[iso_token] = token_id
            self._id_to_token[token_id] = iso_token
            self._isotope_to_id[iso_str] = token_id

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        if self._isotope_list:
            return self._iso_offset + len(self._isotope_list)
        return self._frac_offset + len(self._fraction_list)

    @property
    def pad_idx(self) -> int:
        return PAD_IDX

    @property
    def bos_idx(self) -> int:
        return BOS_IDX

    @property
    def eos_idx(self) -> int:
        return EOS_IDX

    @property
    def unk_idx(self) -> int:
        return UNK_IDX

    @property
    def frac_unk_idx(self) -> int:
        return FRAC_UNK_IDX

    @property
    def n_fraction_tokens(self) -> int:
        """Number of fraction tokens in vocabulary."""
        return len(self._fraction_list)

    @property
    def fraction_token_start(self) -> int:
        """Index of first fraction token."""
        return self._frac_offset

    def is_fraction_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a fraction token."""
        return self._frac_offset <= token_id < self._frac_offset + len(self._fraction_list)

    def is_element_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to an element token."""
        return N_SPECIAL <= token_id < N_SPECIAL + N_ELEMENTS

    def is_integer_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to an integer token."""
        return self._int_offset <= token_id < self._int_offset + MAX_INTEGER

    def is_isotope_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to an isotope token (V14.0)."""
        if not self._isotope_list:
            return False
        return self._iso_offset <= token_id < self._iso_offset + len(self._isotope_list)

    @property
    def iso_unk_idx(self) -> Optional[int]:
        """ISO_UNK token index (None if isotopes not loaded)."""
        return self._iso_unk_idx

    @property
    def n_isotope_tokens(self) -> int:
        """Number of isotope tokens in vocabulary."""
        return len(self._isotope_list)

    @property
    def isotope_token_start(self) -> Optional[int]:
        """Index of first isotope token (None if isotopes not loaded)."""
        if not self._isotope_list:
            return None
        return self._iso_offset

    def isotope_token_to_element(self, token_id: int) -> str:
        """Get the element symbol for an isotope token (e.g., ISO:18O → 'O')."""
        if self.is_isotope_token(token_id):
            iso_str = self._isotope_list[token_id - self._iso_offset]
            # Parse "massSymbol" → symbol is the non-digit suffix
            match = re.match(r'^(\d+)([A-Z][a-z]?)$', iso_str)
            if match:
                return match.group(2)
        raise ValueError(f"Token {token_id} is not an isotope token")

    def isotope_token_to_mass(self, token_id: int) -> int:
        """Get the mass number for an isotope token (e.g., ISO:18O → 18)."""
        if self.is_isotope_token(token_id):
            iso_str = self._isotope_list[token_id - self._iso_offset]
            match = re.match(r'^(\d+)([A-Z][a-z]?)$', iso_str)
            if match:
                return int(match.group(1))
        raise ValueError(f"Token {token_id} is not an isotope token")

    def element_idx_for_isotope(self, token_id: int) -> int:
        """Get the element token index corresponding to an isotope (for embedding init)."""
        elem = self.isotope_token_to_element(token_id)
        return self._token_to_id[elem]

    def fraction_token_to_value(self, token_id: int) -> float:
        """Convert a fraction token ID to its float value."""
        if token_id in self._fraction_to_value:
            return self._fraction_to_value[token_id]
        raise ValueError(f"Token {token_id} is not a fraction token")

    def fraction_token_to_numden(self, token_id: int) -> Tuple[int, int]:
        """Convert a fraction token ID to its (numerator, denominator) pair."""
        if self.is_fraction_token(token_id):
            frac_str = self._fraction_list[token_id - self._frac_offset]
            parts = frac_str.split('/')
            return (int(parts[0]), int(parts[1]))
        raise ValueError(f"Token {token_id} is not a fraction token")

    def encode(self, formula: str, add_bos_eos: bool = True, pad: bool = True) -> List[int]:
        """Tokenize a formula string into token IDs.

        Fractions are GCD-canonicalized during encoding.
        V14.0: Isotope notation {mass}Element is matched as single tokens.
        Returns a list of token IDs, optionally padded to max_len.

        Args:
            formula: Chemical formula string (e.g., "Y1Ba2Cu3{18}O(17/20)")
            add_bos_eos: Whether to add BOS/EOS tokens (default True)
            pad: Whether to pad to max_len (default True)

        Returns:
            List of integer token IDs
        """
        tokens = []

        # V14.0: Use isotope-aware pattern when isotope vocab is loaded
        pattern = _ISOTOPE_FORMULA_PATTERN if self._isotope_list else _FORMULA_PATTERN

        for m in pattern.finditer(formula):
            if self._isotope_list:
                # V14.0: Isotope-aware pattern (6 groups)
                if m.group(1) is not None:
                    # Isotope: {mass}Element
                    mass = m.group(1)
                    elem = m.group(2)
                    iso_str = f"{mass}{elem}"
                    if iso_str in self._isotope_to_id:
                        tokens.append(self._isotope_to_id[iso_str])
                    elif self._iso_unk_idx is not None:
                        tokens.append(self._iso_unk_idx)
                    else:
                        tokens.append(UNK_IDX)
                elif m.group(3) is not None:
                    # Fraction: (num/den)
                    num = int(m.group(3))
                    den = int(m.group(4))
                    g = math.gcd(num, den)
                    num, den = num // g, den // g
                    frac_str = f"{num}/{den}"
                    if frac_str in self._fraction_to_id:
                        tokens.append(self._fraction_to_id[frac_str])
                    else:
                        tokens.append(FRAC_UNK_IDX)
                elif m.group(5) is not None:
                    # Element symbol
                    elem = m.group(5)
                    if elem in self._token_to_id:
                        tokens.append(self._token_to_id[elem])
                    else:
                        tokens.append(UNK_IDX)
                elif m.group(6) is not None:
                    # Integer subscript
                    val = int(m.group(6))
                    if 1 <= val <= MAX_INTEGER:
                        tokens.append(self._token_to_id[str(val)])
                    else:
                        tokens.append(UNK_IDX)
            else:
                # V13.0: Original pattern (4 groups)
                if m.group(1) is not None:
                    # Fraction: (num/den)
                    num = int(m.group(1))
                    den = int(m.group(2))
                    g = math.gcd(num, den)
                    num, den = num // g, den // g
                    frac_str = f"{num}/{den}"
                    if frac_str in self._fraction_to_id:
                        tokens.append(self._fraction_to_id[frac_str])
                    else:
                        tokens.append(FRAC_UNK_IDX)
                elif m.group(3) is not None:
                    # Element symbol
                    elem = m.group(3)
                    if elem in self._token_to_id:
                        tokens.append(self._token_to_id[elem])
                    else:
                        tokens.append(UNK_IDX)
                elif m.group(4) is not None:
                    # Integer subscript
                    val = int(m.group(4))
                    if 1 <= val <= MAX_INTEGER:
                        tokens.append(self._token_to_id[str(val)])
                    else:
                        tokens.append(UNK_IDX)

        if add_bos_eos:
            tokens = [BOS_IDX] + tokens + [EOS_IDX]

        if pad:
            if len(tokens) < self.max_len:
                tokens = tokens + [PAD_IDX] * (self.max_len - len(tokens))
            elif len(tokens) > self.max_len:
                tokens = tokens[:self.max_len - 1] + [EOS_IDX]

        return tokens

    def decode(self, token_ids: List[int], strip_special: bool = True) -> str:
        """Decode token IDs back to a formula string.

        Args:
            token_ids: List of integer token IDs
            strip_special: If True, remove PAD/BOS/EOS from output

        Returns:
            Reconstructed formula string
        """
        parts = []
        for tid in token_ids:
            if strip_special and tid in (PAD_IDX, BOS_IDX, EOS_IDX):
                if tid == EOS_IDX:
                    break  # Stop at EOS
                continue

            if tid == UNK_IDX:
                parts.append('?')
            elif tid == FRAC_UNK_IDX:
                parts.append('(?/?)')
            elif self._iso_unk_idx is not None and tid == self._iso_unk_idx:
                parts.append('{?}?')
            elif tid in self._id_to_token:
                token_str = self._id_to_token[tid]
                if token_str.startswith('FRAC:'):
                    # Convert FRAC:num/den back to (num/den)
                    parts.append(f"({token_str[5:]})")
                elif token_str.startswith('ISO:'):
                    # Convert ISO:massSymbol back to {mass}Symbol
                    iso_str = token_str[4:]  # e.g., "18O"
                    match = re.match(r'^(\d+)([A-Z][a-z]?)$', iso_str)
                    if match:
                        parts.append(f"{{{match.group(1)}}}{match.group(2)}")
                    else:
                        parts.append(f'{{{iso_str}}}')
                else:
                    parts.append(token_str)
            else:
                parts.append('?')

        return ''.join(parts)

    def get_token_name(self, token_id: int) -> str:
        """Get the display name for a token ID."""
        return self._id_to_token.get(token_id, f'<ID:{token_id}>')

    def save(self, path: str):
        """Save tokenizer state to JSON file."""
        version = 'V14.0' if self._isotope_list else 'V13.0'
        state = {
            'version': version,
            'max_len': self.max_len,
            'n_special': N_SPECIAL,
            'n_elements': N_ELEMENTS,
            'max_integer': MAX_INTEGER,
            'fraction_list': self._fraction_list,
            'fraction_vocab_meta': self._fraction_vocab_meta,
            'isotope_list': self._isotope_list,
            'isotope_vocab_meta': self._isotope_vocab_meta,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'FractionAwareTokenizer':
        """Load tokenizer from saved state."""
        with open(path, 'r') as f:
            state = json.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.max_len = state['max_len']
        tokenizer._build_base_vocab()

        # Reconstruct fraction vocab from saved state
        tokenizer._fraction_list = state['fraction_list']
        tokenizer._fraction_to_id = {}
        tokenizer._fraction_to_value = {}
        tokenizer._fraction_vocab_meta = state.get('fraction_vocab_meta', {})

        for i, frac_str in enumerate(tokenizer._fraction_list):
            token_id = tokenizer._frac_offset + i
            frac_token = f"FRAC:{frac_str}"
            tokenizer._token_to_id[frac_token] = token_id
            tokenizer._id_to_token[token_id] = frac_token
            tokenizer._fraction_to_id[frac_str] = token_id
            parts = frac_str.split('/')
            tokenizer._fraction_to_value[token_id] = int(parts[0]) / int(parts[1])

        # V14.0: Reconstruct isotope vocab from saved state
        tokenizer._isotope_list = state.get('isotope_list', [])
        tokenizer._isotope_to_id = {}
        tokenizer._iso_offset = 0
        tokenizer._iso_unk_idx = None
        tokenizer._isotope_vocab_meta = state.get('isotope_vocab_meta', {})

        if tokenizer._isotope_list:
            frac_end = tokenizer._frac_offset + len(tokenizer._fraction_list)
            tokenizer._iso_unk_idx = frac_end
            tokenizer._token_to_id[ISO_UNK_TOKEN] = tokenizer._iso_unk_idx
            tokenizer._id_to_token[tokenizer._iso_unk_idx] = ISO_UNK_TOKEN

            tokenizer._iso_offset = tokenizer._iso_unk_idx + 1
            for i, iso_str in enumerate(tokenizer._isotope_list):
                token_id = tokenizer._iso_offset + i
                iso_token = f"ISO:{iso_str}"
                tokenizer._token_to_id[iso_token] = token_id
                tokenizer._id_to_token[token_id] = iso_token
                tokenizer._isotope_to_id[iso_str] = token_id

        return tokenizer

    def get_old_to_new_token_mapping(self) -> Dict[int, int]:
        """Create a mapping from old (V12) token indices to new (V13) token indices.

        Used by weight migration script to transfer embedding weights.
        Returns a dict mapping old_idx -> new_idx for tokens that exist in both vocabs.
        """
        # Import old vocabulary
        from superconductor.models.autoregressive_decoder import TOKEN_TO_IDX as OLD_TOKEN_TO_IDX

        mapping = {}

        # Map special tokens
        old_special_map = {
            '<PAD>': PAD_IDX,
            '<START>': BOS_IDX,  # Old uses <START>, new uses <BOS>
            '<END>': EOS_IDX,    # Old uses <END>, new uses <EOS>
        }
        for old_tok, new_idx in old_special_map.items():
            if old_tok in OLD_TOKEN_TO_IDX:
                mapping[OLD_TOKEN_TO_IDX[old_tok]] = new_idx

        # Map element tokens
        for elem in ELEMENTS[1:]:
            if elem in OLD_TOKEN_TO_IDX and elem in self._token_to_id:
                mapping[OLD_TOKEN_TO_IDX[elem]] = self._token_to_id[elem]

        # Map digit tokens: old vocab has individual digits '0'-'9'
        # New vocab has integers '1'-'20'. Map '1'-'9' directly.
        for d in range(1, 10):
            old_tok = str(d)
            if old_tok in OLD_TOKEN_TO_IDX and old_tok in self._token_to_id:
                mapping[OLD_TOKEN_TO_IDX[old_tok]] = self._token_to_id[old_tok]

        return mapping

    def get_v13_to_v14_token_mapping(self) -> Dict[int, int]:
        """Create a mapping from V13 token indices to V14 token indices.

        V13→V14 is trivial: all V13 tokens keep their original indices.
        New tokens (ISO_UNK + 291 isotopes) are appended after the last
        fraction token. Returns identity mapping for all V13 tokens.
        """
        # All tokens [0..frac_offset + n_fractions - 1] are unchanged
        v13_vocab_size = self._frac_offset + len(self._fraction_list)
        return {i: i for i in range(v13_vocab_size)}

    def __repr__(self) -> str:
        iso_part = f", n_isotopes={self.n_isotope_tokens}" if self._isotope_list else ""
        return (f"FractionAwareTokenizer(vocab_size={self.vocab_size}, "
                f"n_fractions={self.n_fraction_tokens}{iso_part}, "
                f"max_len={self.max_len})")
