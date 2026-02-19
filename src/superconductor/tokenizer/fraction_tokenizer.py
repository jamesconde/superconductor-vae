"""
FractionAwareTokenizer — V13.0 Semantic Fraction Tokenization.

Vocabulary structure:
    [0] PAD  [1] BOS  [2] EOS  [3] UNK  [4] FRAC_UNK    (5 special tokens)
    [5..122] H, He, Li, ..., Og                            (118 element tokens)
    [123..142] 1, 2, 3, ..., 20                            (20 integer tokens)
    [143..N] FRAC:1/2, FRAC:1/4, FRAC:3/4, ...            (fraction tokens from vocab)

Key features:
    - Single token per fraction: (17/20) → FRAC:17/20 (was 7 tokens)
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


class FractionAwareTokenizer:
    """Semantic fraction tokenizer for superconductor formula generation.

    Replaces digit-by-digit fraction tokenization with single semantic tokens.
    Each fraction (p/q) becomes one FRAC:p/q token instead of multiple character tokens.

    Args:
        fraction_vocab_path: Path to fraction_vocab.json (built by build_fraction_vocab.py)
        max_len: Maximum sequence length (including BOS/EOS)
    """

    def __init__(self, fraction_vocab_path: str = None, max_len: int = 60):
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

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
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
        Returns a list of token IDs, optionally padded to max_len.

        Args:
            formula: Chemical formula string (e.g., "Y1Ba2Cu3O(17/20)")
            add_bos_eos: Whether to add BOS/EOS tokens (default True)
            pad: Whether to pad to max_len (default True)

        Returns:
            List of integer token IDs
        """
        tokens = []

        for m in _FORMULA_PATTERN.finditer(formula):
            if m.group(1) is not None:
                # Fraction: (num/den)
                num = int(m.group(1))
                den = int(m.group(2))
                # GCD canonicalize
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
                    # Integer > 20: use UNK
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
            elif tid in self._id_to_token:
                token_str = self._id_to_token[tid]
                if token_str.startswith('FRAC:'):
                    # Convert FRAC:num/den back to (num/den)
                    parts.append(f"({token_str[5:]})")
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
        state = {
            'version': 'V13.0',
            'max_len': self.max_len,
            'n_special': N_SPECIAL,
            'n_elements': N_ELEMENTS,
            'max_integer': MAX_INTEGER,
            'fraction_list': self._fraction_list,
            'fraction_vocab_meta': self._fraction_vocab_meta,
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

    def __repr__(self) -> str:
        return (f"FractionAwareTokenizer(vocab_size={self.vocab_size}, "
                f"n_fractions={self.n_fraction_tokens}, "
                f"max_len={self.max_len})")
