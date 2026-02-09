"""
N-gram + Structural Hybrid Draft Model for Speculative Decoding (V2).

This module provides fast draft token prediction for chemical formulas
to speed up autoregressive generation via speculative decoding.

Key insight: Chemical formulas have strong structural patterns that can be
exploited for draft token prediction without running the main Transformer:

1. Structural Rules (deterministic):
   - After <START>: always Element
   - After Element: (, Element, digit, or <END>
   - After (: digit only
   - After digit in numerator: digit or /
   - After /: digit only
   - After digit in denominator: digit or )
   - After ): Element or <END>

2. N-gram Statistics (learned):
   - Common element-element transitions (e.g., "Cu" -> "O" common in cuprates)
   - Common denominators (/10, /100, /5, etc.)
   - Frequent trigrams like "/10)", "(1/", "/5)"

3. V2 Improvements (position-aware + chemistry-aware):
   - Position-aware N-grams: P(next | prev2, prev1, position)
   - Element-only context: Skip digits when building context for element prediction
   - Position-aware unigram fallback: P(element | element_position)
   - Chemical family detection: Use first element to constrain predictions

The hybrid model combines structural constraints with n-gram probabilities
to draft k tokens that have ~60-70% acceptance rate with the main model.

Speedup: Each accepted draft token saves one full autoregressive step,
so 3 accepted tokens = 3x speedup for those positions.
"""

import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

# Import tokenizer constants from autoregressive_decoder
from .autoregressive_decoder import (
    TOKEN_TO_IDX, IDX_TO_TOKEN, VOCAB_SIZE,
    PAD_IDX, START_IDX, END_IDX,
    ELEMENTS, DIGITS, SPECIAL_CHARS,
    tokenize_formula
)


# =============================================================================
# Structural State Machine
# =============================================================================

class FormulaState:
    """State machine states for formula parsing."""
    START = 'start'           # At <START> token
    ELEMENT = 'element'       # Just saw an element
    OPEN_PAREN = 'open_paren' # Just saw (
    NUMERATOR = 'numerator'   # In numerator digits
    SLASH = 'slash'           # Just saw /
    DENOMINATOR = 'denominator'  # In denominator digits
    CLOSE_PAREN = 'close_paren'  # Just saw )
    DIGIT = 'digit'           # Saw digit outside fraction (integer stoich)


def _is_element_token(idx: int) -> bool:
    """Check if token index is an element."""
    if idx < 0 or idx >= VOCAB_SIZE:
        return False
    token = IDX_TO_TOKEN.get(idx, '')
    return token in ELEMENTS[1:]  # Skip empty element at index 0


def _is_digit_token(idx: int) -> bool:
    """Check if token index is a digit."""
    if idx < 0 or idx >= VOCAB_SIZE:
        return False
    token = IDX_TO_TOKEN.get(idx, '')
    return token in DIGITS


def _get_token_type(idx: int) -> str:
    """Get the type of a token."""
    if idx == START_IDX:
        return 'start'
    if idx == END_IDX:
        return 'end'
    if idx == PAD_IDX:
        return 'pad'

    token = IDX_TO_TOKEN.get(idx, '')
    if token in ELEMENTS[1:]:
        return 'element'
    if token in DIGITS:
        return 'digit'
    if token == '(':
        return 'open_paren'
    if token == ')':
        return 'close_paren'
    if token == '/':
        return 'slash'
    if token == '.':
        return 'decimal'
    return 'other'


# =============================================================================
# N-gram Draft Model (V2 - Position-Aware + Element-Context)
# =============================================================================

# Chemical families based on first element
CHEMICAL_FAMILIES = {
    # YBCO family (Y-Ba-Cu-O)
    'Y': {'family': 'YBCO', 'likely_elements': ['Ba', 'Cu', 'O', 'Ca', 'Pr']},
    # LSCO family (La-Sr-Cu-O)
    'La': {'family': 'LSCO', 'likely_elements': ['Sr', 'Cu', 'O', 'Ba', 'Ca']},
    # Bi-cuprate family (Bi-Sr-Ca-Cu-O)
    'Bi': {'family': 'Bi-cuprate', 'likely_elements': ['Sr', 'Ca', 'Cu', 'O', 'Pb']},
    # Tl-cuprate family (Tl-Ba-Ca-Cu-O)
    'Tl': {'family': 'Tl-cuprate', 'likely_elements': ['Ba', 'Ca', 'Cu', 'O']},
    # Hg-cuprate family (Hg-Ba-Ca-Cu-O)
    'Hg': {'family': 'Hg-cuprate', 'likely_elements': ['Ba', 'Ca', 'Cu', 'O']},
    # Iron-based family (Fe-As/P/Se)
    'Fe': {'family': 'Iron-based', 'likely_elements': ['As', 'P', 'Se', 'O', 'La', 'Ba', 'K']},
    'Ba': {'family': 'Iron-based-122', 'likely_elements': ['Fe', 'As', 'K', 'Co', 'Cu', 'O']},
    'Sr': {'family': 'Sr-based', 'likely_elements': ['Cu', 'O', 'La', 'Ca', 'Fe']},
    # MgB2 family
    'Mg': {'family': 'MgB2', 'likely_elements': ['B', 'Al', 'C']},
    # Conventional superconductors
    'Nb': {'family': 'Conventional', 'likely_elements': ['Ti', 'Sn', 'N', 'O', 'Ge']},
    'Pb': {'family': 'Conventional', 'likely_elements': ['Bi', 'Sn', 'In', 'O']},
}


class NGramDraft:
    """
    Position-aware trigram model for chemical formulas (V2).

    V2 improvements over V1:
    1. Position-aware trigrams: P(next | prev2, prev1, position)
    2. Element-only context: Skip digits when predicting elements
    3. Position-aware element unigrams: P(element | element_position)
    4. Chemical family hints from first element

    Falls back through: position-trigram → trigram → position-bigram → bigram →
                       position-unigram → unigram
    """

    def __init__(self, n: int = 3, smoothing: float = 0.1):
        """
        Args:
            n: N-gram order (default 3 for trigram)
            smoothing: Laplace smoothing factor
        """
        self.n = n
        self.smoothing = smoothing

        # Standard N-gram counts (fallback)
        self.unigram_counts: Dict[int, int] = defaultdict(int)
        self.bigram_counts: Dict[Tuple[int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.trigram_counts: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # V2: Position-aware N-gram counts
        # Key: (prev2_idx, prev1_idx, position) or (prev1_idx, position) or (position,)
        self.pos_trigram_counts: Dict[Tuple[int, int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.pos_bigram_counts: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.pos_unigram_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # V2: Element-only context (skip digits) for element prediction
        # Key: (prev_elem2, prev_elem1, elem_position) → element
        self.elem_trigram_counts: Dict[Tuple[int, int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.elem_bigram_counts: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # V2: Position-aware element unigrams: P(element | element_position)
        self.elem_position_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # V2: Family-specific element transitions
        self.family_element_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Total counts for normalization
        self.total_unigram = 0
        self.bigram_totals: Dict[Tuple[int], int] = defaultdict(int)
        self.trigram_totals: Dict[Tuple[int, int], int] = defaultdict(int)

        # V2: Position-aware totals
        self.pos_trigram_totals: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.pos_bigram_totals: Dict[Tuple[int, int], int] = defaultdict(int)
        self.pos_unigram_totals: Dict[int, int] = defaultdict(int)
        self.elem_trigram_totals: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.elem_bigram_totals: Dict[Tuple[int, int], int] = defaultdict(int)
        self.elem_position_totals: Dict[int, int] = defaultdict(int)
        self.family_totals: Dict[str, int] = defaultdict(int)

        # Cached most likely next tokens (standard)
        self._trigram_argmax: Dict[Tuple[int, int], int] = {}
        self._bigram_argmax: Dict[Tuple[int], int] = {}
        self._unigram_argmax: int = -1

        # V2: Cached argmax for position-aware
        self._pos_trigram_argmax: Dict[Tuple[int, int, int], int] = {}
        self._pos_bigram_argmax: Dict[Tuple[int, int], int] = {}
        self._pos_unigram_argmax: Dict[int, int] = {}
        self._elem_trigram_argmax: Dict[Tuple[int, int, int], int] = {}
        self._elem_bigram_argmax: Dict[Tuple[int, int], int] = {}
        self._elem_position_argmax: Dict[int, int] = {}

        # Build element indices set for fast lookup
        self._element_indices: Set[int] = set()
        for elem in ELEMENTS[1:]:  # Skip empty element
            if elem in TOKEN_TO_IDX:
                self._element_indices.add(TOKEN_TO_IDX[elem])

    def _extract_element_context(self, indices: List[int]) -> List[int]:
        """Extract only element tokens from sequence (skip digits, parens, etc.)."""
        return [idx for idx in indices if idx in self._element_indices]

    def _get_family(self, indices: List[int]) -> Optional[str]:
        """Detect chemical family from first element."""
        elements = self._extract_element_context(indices)
        if elements:
            first_elem = IDX_TO_TOKEN.get(elements[0], '')
            if first_elem in CHEMICAL_FAMILIES:
                return CHEMICAL_FAMILIES[first_elem]['family']
        return None

    def _get_family_likely_elements(self, indices: List[int]) -> Set[int]:
        """Get likely next elements based on chemical family."""
        elements = self._extract_element_context(indices)
        if elements:
            first_elem = IDX_TO_TOKEN.get(elements[0], '')
            if first_elem in CHEMICAL_FAMILIES:
                likely = CHEMICAL_FAMILIES[first_elem]['likely_elements']
                return {TOKEN_TO_IDX[e] for e in likely if e in TOKEN_TO_IDX}
        return set()

    def train(self, formulas: List[str], max_len: int = 60):
        """
        Train n-gram model on list of formulas (V2: position-aware).

        Args:
            formulas: List of formula strings
            max_len: Maximum sequence length for tokenization
        """
        print(f"Training V2 position-aware {self.n}-gram model on {len(formulas)} formulas...")

        for formula in formulas:
            # Tokenize formula
            tokens = tokenize_formula(formula)

            # Convert to indices with START and END
            indices = [START_IDX]
            for token in tokens:
                if token in TOKEN_TO_IDX:
                    indices.append(TOKEN_TO_IDX[token])
            indices.append(END_IDX)

            # Limit length
            if len(indices) > max_len:
                indices = indices[:max_len-1] + [END_IDX]

            # Track element positions separately
            elem_positions = []  # (idx, elem_pos) for each element
            elem_pos = 0
            for i, idx in enumerate(indices):
                if idx in self._element_indices:
                    elem_positions.append((i, elem_pos))
                    elem_pos += 1

            # Detect family
            family = self._get_family(indices)

            # Count n-grams
            for i, idx in enumerate(indices):
                # === Standard n-grams (fallback) ===
                # Unigram
                self.unigram_counts[idx] += 1
                self.total_unigram += 1

                # Bigram
                if i >= 1:
                    context = (indices[i-1],)
                    self.bigram_counts[context][idx] += 1
                    self.bigram_totals[context] += 1

                # Trigram
                if i >= 2:
                    context = (indices[i-2], indices[i-1])
                    self.trigram_counts[context][idx] += 1
                    self.trigram_totals[context] += 1

                # === V2: Position-aware n-grams ===
                # Position-aware unigram
                self.pos_unigram_counts[i][idx] += 1
                self.pos_unigram_totals[i] += 1

                # Position-aware bigram
                if i >= 1:
                    pos_ctx = (indices[i-1], i)
                    self.pos_bigram_counts[pos_ctx][idx] += 1
                    self.pos_bigram_totals[pos_ctx] += 1

                # Position-aware trigram
                if i >= 2:
                    pos_ctx = (indices[i-2], indices[i-1], i)
                    self.pos_trigram_counts[pos_ctx][idx] += 1
                    self.pos_trigram_totals[pos_ctx] += 1

            # === V2: Element-only context for element prediction ===
            elements_only = self._extract_element_context(indices)
            for elem_i, elem_idx in enumerate(elements_only):
                # Element position unigram: P(element | elem_position)
                self.elem_position_counts[elem_i][elem_idx] += 1
                self.elem_position_totals[elem_i] += 1

                # Element bigram with position
                if elem_i >= 1:
                    elem_ctx = (elements_only[elem_i-1], elem_i)
                    self.elem_bigram_counts[elem_ctx][elem_idx] += 1
                    self.elem_bigram_totals[elem_ctx] += 1

                # Element trigram with position
                if elem_i >= 2:
                    elem_ctx = (elements_only[elem_i-2], elements_only[elem_i-1], elem_i)
                    self.elem_trigram_counts[elem_ctx][elem_idx] += 1
                    self.elem_trigram_totals[elem_ctx] += 1

                # Family-specific element counts
                if family:
                    self.family_element_counts[family][elem_idx] += 1
                    self.family_totals[family] += 1

        # Compute argmax caches
        self._compute_argmax_caches()

        print(f"  Unigram vocab: {len(self.unigram_counts)} tokens")
        print(f"  Bigram contexts: {len(self.bigram_counts)}")
        print(f"  Trigram contexts: {len(self.trigram_counts)}")
        print(f"  V2 Position-aware trigram contexts: {len(self.pos_trigram_counts)}")
        print(f"  V2 Element-position contexts: {len(self.elem_position_counts)}")
        print(f"  V2 Element trigram contexts: {len(self.elem_trigram_counts)}")
        print(f"  V2 Families detected: {list(self.family_totals.keys())}")

    def _compute_argmax_caches(self):
        """Pre-compute most likely next token for each context."""
        # Standard trigram argmax
        for context, token_counts in self.trigram_counts.items():
            if token_counts:
                self._trigram_argmax[context] = max(token_counts, key=token_counts.get)

        # Standard bigram argmax
        for context, token_counts in self.bigram_counts.items():
            if token_counts:
                self._bigram_argmax[context] = max(token_counts, key=token_counts.get)

        # Standard unigram argmax
        if self.unigram_counts:
            self._unigram_argmax = max(self.unigram_counts, key=self.unigram_counts.get)

        # V2: Position-aware argmax caches
        for context, token_counts in self.pos_trigram_counts.items():
            if token_counts:
                self._pos_trigram_argmax[context] = max(token_counts, key=token_counts.get)

        for context, token_counts in self.pos_bigram_counts.items():
            if token_counts:
                self._pos_bigram_argmax[context] = max(token_counts, key=token_counts.get)

        for pos, token_counts in self.pos_unigram_counts.items():
            if token_counts:
                self._pos_unigram_argmax[pos] = max(token_counts, key=token_counts.get)

        # V2: Element-specific argmax
        for context, token_counts in self.elem_trigram_counts.items():
            if token_counts:
                self._elem_trigram_argmax[context] = max(token_counts, key=token_counts.get)

        for context, token_counts in self.elem_bigram_counts.items():
            if token_counts:
                self._elem_bigram_argmax[context] = max(token_counts, key=token_counts.get)

        for pos, token_counts in self.elem_position_counts.items():
            if token_counts:
                self._elem_position_argmax[pos] = max(token_counts, key=token_counts.get)

    def predict_next(self, context: List[int], position: Optional[int] = None) -> Optional[int]:
        """
        Predict most likely next token given context (V2: position-aware).

        Args:
            context: List of previous token indices
            position: Current position in sequence (if known)

        Returns:
            Most likely next token index, or None if no prediction
        """
        if position is None:
            position = len(context)

        # V2: Try position-aware trigram first
        if len(context) >= 2:
            pos_ctx = (context[-2], context[-1], position)
            if pos_ctx in self._pos_trigram_argmax:
                return self._pos_trigram_argmax[pos_ctx]

        # Fallback to standard trigram
        if len(context) >= 2:
            trigram_ctx = (context[-2], context[-1])
            if trigram_ctx in self._trigram_argmax:
                return self._trigram_argmax[trigram_ctx]

        # V2: Try position-aware bigram
        if len(context) >= 1:
            pos_ctx = (context[-1], position)
            if pos_ctx in self._pos_bigram_argmax:
                return self._pos_bigram_argmax[pos_ctx]

        # Fallback to standard bigram
        if len(context) >= 1:
            bigram_ctx = (context[-1],)
            if bigram_ctx in self._bigram_argmax:
                return self._bigram_argmax[bigram_ctx]

        # V2: Try position-aware unigram
        if position in self._pos_unigram_argmax:
            return self._pos_unigram_argmax[position]

        # Fallback to standard unigram
        if self._unigram_argmax >= 0:
            return self._unigram_argmax

        return None

    def predict_next_element(
        self,
        context: List[int],
        elem_position: int,
        valid_elements: Optional[Set[int]] = None,
    ) -> Optional[int]:
        """
        Predict most likely next element using element-only context (V2).

        Args:
            context: Full token sequence (elements will be extracted)
            elem_position: Which element position we're predicting (0-indexed)
            valid_elements: Optional set of valid element indices

        Returns:
            Most likely element index, or None
        """
        elements = self._extract_element_context(context)

        # V2: Try element trigram with position
        if len(elements) >= 2:
            elem_ctx = (elements[-2], elements[-1], elem_position)
            if elem_ctx in self._elem_trigram_argmax:
                pred = self._elem_trigram_argmax[elem_ctx]
                if valid_elements is None or pred in valid_elements:
                    return pred

        # V2: Try element bigram with position
        if len(elements) >= 1:
            elem_ctx = (elements[-1], elem_position)
            if elem_ctx in self._elem_bigram_argmax:
                pred = self._elem_bigram_argmax[elem_ctx]
                if valid_elements is None or pred in valid_elements:
                    return pred

        # V2: Try element position unigram
        if elem_position in self._elem_position_argmax:
            pred = self._elem_position_argmax[elem_position]
            if valid_elements is None or pred in valid_elements:
                return pred

        # Fallback: try family hints
        family_elems = self._get_family_likely_elements(context)
        if family_elems:
            if valid_elements:
                family_elems = family_elems & valid_elements
            if family_elems:
                # Return most common in family
                family = self._get_family(context)
                if family and family in self.family_element_counts:
                    counts = self.family_element_counts[family]
                    valid_counts = {k: v for k, v in counts.items() if k in family_elems}
                    if valid_counts:
                        return max(valid_counts, key=valid_counts.get)

        return None

    def get_top_k(
        self,
        context: List[int],
        k: int = 5,
        position: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most likely next tokens with probabilities (V2: position-aware).

        Args:
            context: List of previous token indices
            k: Number of top tokens to return
            position: Current position (optional)

        Returns:
            List of (token_idx, probability) tuples
        """
        if position is None:
            position = len(context)

        # V2: Try position-aware trigram first
        if len(context) >= 2:
            pos_ctx = (context[-2], context[-1], position)
            if pos_ctx in self.pos_trigram_counts:
                counts = self.pos_trigram_counts[pos_ctx]
                total = self.pos_trigram_totals[pos_ctx]
                if total > 0:
                    items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                    return [(idx, count / total) for idx, count in items]

        # Fallback to standard trigram
        if len(context) >= 2:
            trigram_ctx = (context[-2], context[-1])
            if trigram_ctx in self.trigram_counts:
                counts = self.trigram_counts[trigram_ctx]
                total = self.trigram_totals[trigram_ctx]
                if total > 0:
                    items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                    return [(idx, count / total) for idx, count in items]

        # V2: Try position-aware bigram
        if len(context) >= 1:
            pos_ctx = (context[-1], position)
            if pos_ctx in self.pos_bigram_counts:
                counts = self.pos_bigram_counts[pos_ctx]
                total = self.pos_bigram_totals[pos_ctx]
                if total > 0:
                    items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                    return [(idx, count / total) for idx, count in items]

        # Fallback to standard bigram
        if len(context) >= 1:
            bigram_ctx = (context[-1],)
            if bigram_ctx in self.bigram_counts:
                counts = self.bigram_counts[bigram_ctx]
                total = self.bigram_totals[bigram_ctx]
                if total > 0:
                    items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                    return [(idx, count / total) for idx, count in items]

        # V2: Try position-aware unigram
        if position in self.pos_unigram_counts:
            counts = self.pos_unigram_counts[position]
            total = self.pos_unigram_totals[position]
            if total > 0:
                items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                return [(idx, count / total) for idx, count in items]

        # Fallback to standard unigram
        if self.total_unigram > 0:
            items = sorted(self.unigram_counts.items(), key=lambda x: -x[1])[:k]
            return [(idx, count / self.total_unigram) for idx, count in items]

        return []

    def get_top_k_elements(
        self,
        context: List[int],
        k: int = 5,
        elem_position: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most likely next elements using element-only context (V2).

        Args:
            context: Full token sequence
            k: Number of top elements to return
            elem_position: Element position (optional, computed if not given)

        Returns:
            List of (element_idx, probability) tuples
        """
        elements = self._extract_element_context(context)
        if elem_position is None:
            elem_position = len(elements)

        # V2: Try element trigram with position
        if len(elements) >= 2:
            elem_ctx = (elements[-2], elements[-1], elem_position)
            if elem_ctx in self.elem_trigram_counts:
                counts = self.elem_trigram_counts[elem_ctx]
                total = self.elem_trigram_totals[elem_ctx]
                if total > 0:
                    items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                    return [(idx, count / total) for idx, count in items]

        # V2: Try element bigram with position
        if len(elements) >= 1:
            elem_ctx = (elements[-1], elem_position)
            if elem_ctx in self.elem_bigram_counts:
                counts = self.elem_bigram_counts[elem_ctx]
                total = self.elem_bigram_totals[elem_ctx]
                if total > 0:
                    items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                    return [(idx, count / total) for idx, count in items]

        # V2: Try element position unigram
        if elem_position in self.elem_position_counts:
            counts = self.elem_position_counts[elem_position]
            total = self.elem_position_totals[elem_position]
            if total > 0:
                items = sorted(counts.items(), key=lambda x: -x[1])[:k]
                return [(idx, count / total) for idx, count in items]

        return []

    def draft_k_tokens(self, tokens: List[int], k: int = 5) -> List[int]:
        """
        Draft k tokens using n-gram predictions (V2: position-aware).

        Args:
            tokens: Current token sequence
            k: Number of tokens to draft

        Returns:
            List of k drafted token indices
        """
        context = list(tokens)
        drafted = []
        position = len(tokens)

        for _ in range(k):
            next_token = self.predict_next(context, position=position)
            if next_token is None or next_token == END_IDX:
                break
            drafted.append(next_token)
            context.append(next_token)
            position += 1

        # Pad with END if needed
        while len(drafted) < k:
            drafted.append(END_IDX)

        return drafted[:k]

    def save(self, path: Path):
        """Save n-gram model to pickle file (V2: includes position-aware data)."""
        data = {
            'version': 2,  # V2 format marker
            'n': self.n,
            'smoothing': self.smoothing,
            # Standard n-grams
            'unigram_counts': dict(self.unigram_counts),
            'bigram_counts': {k: dict(v) for k, v in self.bigram_counts.items()},
            'trigram_counts': {k: dict(v) for k, v in self.trigram_counts.items()},
            'total_unigram': self.total_unigram,
            'bigram_totals': dict(self.bigram_totals),
            'trigram_totals': dict(self.trigram_totals),
            # V2: Position-aware n-grams
            'pos_trigram_counts': {k: dict(v) for k, v in self.pos_trigram_counts.items()},
            'pos_bigram_counts': {k: dict(v) for k, v in self.pos_bigram_counts.items()},
            'pos_unigram_counts': {k: dict(v) for k, v in self.pos_unigram_counts.items()},
            'pos_trigram_totals': dict(self.pos_trigram_totals),
            'pos_bigram_totals': dict(self.pos_bigram_totals),
            'pos_unigram_totals': dict(self.pos_unigram_totals),
            # V2: Element-only context
            'elem_trigram_counts': {k: dict(v) for k, v in self.elem_trigram_counts.items()},
            'elem_bigram_counts': {k: dict(v) for k, v in self.elem_bigram_counts.items()},
            'elem_position_counts': {k: dict(v) for k, v in self.elem_position_counts.items()},
            'elem_trigram_totals': dict(self.elem_trigram_totals),
            'elem_bigram_totals': dict(self.elem_bigram_totals),
            'elem_position_totals': dict(self.elem_position_totals),
            # V2: Family-specific
            'family_element_counts': {k: dict(v) for k, v in self.family_element_counts.items()},
            'family_totals': dict(self.family_totals),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved V2 n-gram model to {path}")

    @classmethod
    def load(cls, path: Path) -> 'NGramDraft':
        """Load n-gram model from pickle file (supports V1 and V2)."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(n=data['n'], smoothing=data['smoothing'])

        # Standard n-grams (V1 compatible)
        model.unigram_counts = defaultdict(int, data['unigram_counts'])
        model.bigram_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data['bigram_counts'].items():
            model.bigram_counts[k] = defaultdict(int, v)
        model.trigram_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data['trigram_counts'].items():
            model.trigram_counts[k] = defaultdict(int, v)
        model.total_unigram = data['total_unigram']
        model.bigram_totals = defaultdict(int, data['bigram_totals'])
        model.trigram_totals = defaultdict(int, data['trigram_totals'])

        # V2: Position-aware n-grams (if present)
        if data.get('version', 1) >= 2:
            for k, v in data.get('pos_trigram_counts', {}).items():
                model.pos_trigram_counts[k] = defaultdict(int, v)
            for k, v in data.get('pos_bigram_counts', {}).items():
                model.pos_bigram_counts[k] = defaultdict(int, v)
            for k, v in data.get('pos_unigram_counts', {}).items():
                model.pos_unigram_counts[k] = defaultdict(int, v)
            model.pos_trigram_totals = defaultdict(int, data.get('pos_trigram_totals', {}))
            model.pos_bigram_totals = defaultdict(int, data.get('pos_bigram_totals', {}))
            model.pos_unigram_totals = defaultdict(int, data.get('pos_unigram_totals', {}))

            # V2: Element-only context
            for k, v in data.get('elem_trigram_counts', {}).items():
                model.elem_trigram_counts[k] = defaultdict(int, v)
            for k, v in data.get('elem_bigram_counts', {}).items():
                model.elem_bigram_counts[k] = defaultdict(int, v)
            for k, v in data.get('elem_position_counts', {}).items():
                model.elem_position_counts[k] = defaultdict(int, v)
            model.elem_trigram_totals = defaultdict(int, data.get('elem_trigram_totals', {}))
            model.elem_bigram_totals = defaultdict(int, data.get('elem_bigram_totals', {}))
            model.elem_position_totals = defaultdict(int, data.get('elem_position_totals', {}))

            # V2: Family-specific
            for k, v in data.get('family_element_counts', {}).items():
                model.family_element_counts[k] = defaultdict(int, v)
            model.family_totals = defaultdict(int, data.get('family_totals', {}))

        model._compute_argmax_caches()
        version = data.get('version', 1)
        print(f"Loaded V{version} n-gram model from {path}")
        return model


# =============================================================================
# Structural Draft Model (V2 - Position-Aware)
# =============================================================================

class StructuralDraft:
    """
    Structure-aware draft model using formula grammar (V2: position-aware).

    V2 improvements:
    1. Position-aware element distributions: P(element | element_position)
    2. Better denominator completion using partial match
    3. Family-aware element weighting

    Uses a state machine to predict valid next tokens based on formula structure,
    combined with learned element and digit frequency distributions.
    """

    def __init__(self):
        """Initialize with default distributions (can be trained)."""
        # Element frequency from training (global fallback)
        self.element_probs: Dict[int, float] = {}
        # V2: Position-aware element frequency: position -> element -> prob
        self.position_element_probs: Dict[int, Dict[int, float]] = {}
        # Digit frequency from training (for denominators especially)
        self.digit_probs: Dict[int, float] = {}
        # Common denominator patterns (e.g., /10, /100, /5)
        self.denominator_probs: Dict[str, float] = {}
        # V2: Numerator patterns (e.g., 1, 2, 3 for simple fractions)
        self.numerator_probs: Dict[str, float] = {}

        # Build element indices set for fast lookup
        self._element_indices: Set[int] = set()
        for elem in ELEMENTS[1:]:  # Skip empty element
            if elem in TOKEN_TO_IDX:
                self._element_indices.add(TOKEN_TO_IDX[elem])

        # Build digit indices
        self._digit_indices: Set[int] = set()
        for d in DIGITS:
            if d in TOKEN_TO_IDX:
                self._digit_indices.add(TOKEN_TO_IDX[d])

        # Special tokens
        self._open_paren_idx = TOKEN_TO_IDX.get('(', -1)
        self._close_paren_idx = TOKEN_TO_IDX.get(')', -1)
        self._slash_idx = TOKEN_TO_IDX.get('/', -1)

    def train(self, formulas: List[str]):
        """
        Learn element and digit distributions from training data (V2: position-aware).

        Args:
            formulas: List of formula strings
        """
        print(f"Training V2 structural model on {len(formulas)} formulas...")

        element_counts: Dict[int, int] = defaultdict(int)
        # V2: Position-aware element counts
        position_element_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        position_totals: Dict[int, int] = defaultdict(int)

        digit_counts: Dict[int, int] = defaultdict(int)
        denominator_counts: Dict[str, int] = defaultdict(int)
        numerator_counts: Dict[str, int] = defaultdict(int)

        for formula in formulas:
            tokens = tokenize_formula(formula)

            # Find fractions (numerator/denominator)
            formula_str = formula
            fraction_matches = re.findall(r'(\d+)/(\d+)', formula_str)
            for numer, denom in fraction_matches:
                numerator_counts[numer] += 1
                denominator_counts[denom] += 1

            # Track element position
            elem_position = 0
            for token in tokens:
                if token in TOKEN_TO_IDX:
                    idx = TOKEN_TO_IDX[token]
                    if token in ELEMENTS[1:]:
                        element_counts[idx] += 1
                        # V2: Track by position
                        position_element_counts[elem_position][idx] += 1
                        position_totals[elem_position] += 1
                        elem_position += 1
                    elif token in DIGITS:
                        digit_counts[idx] += 1

        # Convert to probabilities
        total_elements = sum(element_counts.values())
        if total_elements > 0:
            self.element_probs = {k: v / total_elements for k, v in element_counts.items()}

        # V2: Position-aware element probabilities
        for pos, counts in position_element_counts.items():
            total = position_totals[pos]
            if total > 0:
                self.position_element_probs[pos] = {k: v / total for k, v in counts.items()}

        total_digits = sum(digit_counts.values())
        if total_digits > 0:
            self.digit_probs = {k: v / total_digits for k, v in digit_counts.items()}

        total_denoms = sum(denominator_counts.values())
        if total_denoms > 0:
            self.denominator_probs = {k: v / total_denoms for k, v in denominator_counts.items()}

        total_numers = sum(numerator_counts.values())
        if total_numers > 0:
            self.numerator_probs = {k: v / total_numers for k, v in numerator_counts.items()}

        print(f"  Learned {len(self.element_probs)} element distributions")
        print(f"  V2 Position-aware element distributions: {len(self.position_element_probs)} positions")
        print(f"  Top 5 elements: {self._get_top_elements(5)}")
        print(f"  Top 5 denominators: {list(sorted(denominator_counts.items(), key=lambda x: -x[1]))[:5]}")
        print(f"  Top 5 numerators: {list(sorted(numerator_counts.items(), key=lambda x: -x[1]))[:5]}")

    def _get_top_elements(self, k: int) -> List[str]:
        """Get top k most frequent elements."""
        sorted_items = sorted(self.element_probs.items(), key=lambda x: -x[1])[:k]
        return [IDX_TO_TOKEN[idx] for idx, _ in sorted_items]

    def _get_top_elements_at_position(self, position: int, k: int) -> List[str]:
        """Get top k most frequent elements at a specific position (V2)."""
        if position in self.position_element_probs:
            sorted_items = sorted(self.position_element_probs[position].items(), key=lambda x: -x[1])[:k]
            return [IDX_TO_TOKEN[idx] for idx, _ in sorted_items]
        return self._get_top_elements(k)

    def _count_elements(self, tokens: List[int]) -> int:
        """Count number of elements in token sequence."""
        return sum(1 for idx in tokens if idx in self._element_indices)

    def parse_state(self, tokens: List[int]) -> Tuple[str, dict]:
        """
        Parse token sequence to determine current formula state (V2: tracks element position).

        Args:
            tokens: List of token indices

        Returns:
            (state_name, state_info) where state_info contains context like
            current numerator, paren depth, element position, etc.
        """
        if not tokens:
            return FormulaState.START, {'element_position': 0}

        # Track state through the sequence
        state = FormulaState.START
        paren_depth = 0
        in_fraction = False
        numerator_digits = []
        denominator_digits = []
        element_position = 0  # V2: Track which element we're on

        for idx in tokens:
            token_type = _get_token_type(idx)

            if token_type == 'start':
                state = FormulaState.START
            elif token_type == 'element':
                state = FormulaState.ELEMENT
                in_fraction = False
                numerator_digits = []
                denominator_digits = []
                element_position += 1  # V2: Increment after seeing element
            elif token_type == 'open_paren':
                state = FormulaState.OPEN_PAREN
                paren_depth += 1
                in_fraction = True
                numerator_digits = []
                denominator_digits = []
            elif token_type == 'digit':
                if state == FormulaState.OPEN_PAREN or state == FormulaState.NUMERATOR:
                    state = FormulaState.NUMERATOR
                    numerator_digits.append(IDX_TO_TOKEN[idx])
                elif state == FormulaState.SLASH or state == FormulaState.DENOMINATOR:
                    state = FormulaState.DENOMINATOR
                    denominator_digits.append(IDX_TO_TOKEN[idx])
                else:
                    # Integer stoichiometry (not in fraction)
                    state = FormulaState.DIGIT
            elif token_type == 'slash':
                state = FormulaState.SLASH
            elif token_type == 'close_paren':
                state = FormulaState.CLOSE_PAREN
                paren_depth -= 1
                in_fraction = False
            elif token_type == 'end':
                break

        info = {
            'paren_depth': paren_depth,
            'in_fraction': in_fraction,
            'numerator': ''.join(numerator_digits) if numerator_digits else None,
            'denominator': ''.join(denominator_digits) if denominator_digits else None,
            'element_position': element_position,  # V2: Next element position
        }

        return state, info

    def get_valid_next_tokens(self, state: str, info: dict) -> Set[int]:
        """
        Get set of valid next token indices given current state.

        Args:
            state: Current formula state
            info: State information dict

        Returns:
            Set of valid token indices
        """
        valid = set()

        if state == FormulaState.START:
            # After <START>: only elements allowed
            valid = self._element_indices.copy()

        elif state == FormulaState.ELEMENT:
            # After element: (, digit, element, or END
            valid = self._element_indices.copy()
            valid.add(self._open_paren_idx)
            valid.update(self._digit_indices)
            valid.add(END_IDX)

        elif state == FormulaState.OPEN_PAREN:
            # After (: only digits for numerator
            valid = self._digit_indices.copy()

        elif state == FormulaState.NUMERATOR:
            # In numerator: digit or /
            valid = self._digit_indices.copy()
            valid.add(self._slash_idx)

        elif state == FormulaState.SLASH:
            # After /: only digits for denominator
            valid = self._digit_indices.copy()

        elif state == FormulaState.DENOMINATOR:
            # In denominator: digit or )
            valid = self._digit_indices.copy()
            valid.add(self._close_paren_idx)

        elif state == FormulaState.CLOSE_PAREN:
            # After ): element or END
            valid = self._element_indices.copy()
            valid.add(END_IDX)

        elif state == FormulaState.DIGIT:
            # After integer digit (outside fraction): digit, element, or END
            valid = self._digit_indices.copy()
            valid.update(self._element_indices)
            valid.add(END_IDX)

        # Remove invalid indices
        valid.discard(-1)

        return valid

    def predict_next(self, tokens: List[int]) -> Optional[int]:
        """
        Predict most likely valid next token (V2: position-aware).

        Args:
            tokens: Current token sequence

        Returns:
            Most likely valid token index, or None
        """
        state, info = self.parse_state(tokens)
        valid_tokens = self.get_valid_next_tokens(state, info)

        if not valid_tokens:
            return None

        # V2: Get element position for position-aware prediction
        elem_position = info.get('element_position', 0)

        # Pick most likely from valid set based on learned distributions
        if state in [FormulaState.START, FormulaState.ELEMENT,
                     FormulaState.CLOSE_PAREN, FormulaState.DIGIT]:
            # Might output element - use position-aware element distribution
            element_tokens = valid_tokens & self._element_indices
            if element_tokens:
                # V2: Try position-aware distribution first
                if elem_position in self.position_element_probs:
                    pos_probs = self.position_element_probs[elem_position]
                    # Filter to valid elements
                    valid_probs = {k: v for k, v in pos_probs.items() if k in element_tokens}
                    if valid_probs:
                        best = max(valid_probs, key=valid_probs.get)
                        return best

                # Fallback to global element distribution
                if self.element_probs:
                    best = max(element_tokens, key=lambda x: self.element_probs.get(x, 0))
                    return best

        if state in [FormulaState.OPEN_PAREN, FormulaState.NUMERATOR,
                     FormulaState.SLASH, FormulaState.DENOMINATOR]:
            # Might output digit
            digit_tokens = valid_tokens & self._digit_indices
            if digit_tokens:
                # V2: For numerators, try to complete common patterns
                if state == FormulaState.NUMERATOR and info.get('numerator'):
                    partial = info['numerator']
                    for numer, prob in sorted(self.numerator_probs.items(),
                                               key=lambda x: -x[1]):
                        if numer.startswith(partial) and len(numer) > len(partial):
                            next_digit = numer[len(partial)]
                            if next_digit in TOKEN_TO_IDX:
                                return TOKEN_TO_IDX[next_digit]

                # For denominators, try to complete common patterns
                if state == FormulaState.DENOMINATOR and info.get('denominator'):
                    partial = info['denominator']
                    for denom, prob in sorted(self.denominator_probs.items(),
                                               key=lambda x: -x[1]):
                        if denom.startswith(partial) and len(denom) > len(partial):
                            next_digit = denom[len(partial)]
                            if next_digit in TOKEN_TO_IDX:
                                return TOKEN_TO_IDX[next_digit]

                # Otherwise pick by digit frequency
                if self.digit_probs:
                    best = max(digit_tokens, key=lambda x: self.digit_probs.get(x, 0))
                    return best

        # Default: pick first valid
        return next(iter(valid_tokens)) if valid_tokens else None

    def draft_k_tokens(self, tokens: List[int], k: int = 5) -> List[int]:
        """
        Draft k tokens using structural predictions.

        Args:
            tokens: Current token sequence
            k: Number of tokens to draft

        Returns:
            List of k drafted token indices
        """
        context = list(tokens)
        drafted = []

        for _ in range(k):
            next_token = self.predict_next(context)
            if next_token is None or next_token == END_IDX:
                drafted.append(END_IDX)
                break
            drafted.append(next_token)
            context.append(next_token)

        # Pad with END if needed
        while len(drafted) < k:
            drafted.append(END_IDX)

        return drafted[:k]

    def save(self, path: Path):
        """Save structural model to pickle file (V2: includes position-aware data)."""
        data = {
            'version': 2,  # V2 format marker
            'element_probs': self.element_probs,
            'digit_probs': self.digit_probs,
            'denominator_probs': self.denominator_probs,
            # V2 additions
            'position_element_probs': dict(self.position_element_probs),
            'numerator_probs': self.numerator_probs,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved V2 structural model to {path}")

    @classmethod
    def load(cls, path: Path) -> 'StructuralDraft':
        """Load structural model from pickle file (supports V1 and V2)."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls()
        model.element_probs = data['element_probs']
        model.digit_probs = data['digit_probs']
        model.denominator_probs = data['denominator_probs']

        # V2 additions (if present)
        if data.get('version', 1) >= 2:
            model.position_element_probs = data.get('position_element_probs', {})
            model.numerator_probs = data.get('numerator_probs', {})

        version = data.get('version', 1)
        print(f"Loaded V{version} structural model from {path}")
        return model


# =============================================================================
# Hybrid Draft Model (V2 - Position-Aware + Element-Context)
# =============================================================================

class HybridDraft:
    """
    Combined N-gram + Structural draft model (V2: position-aware + element-context).

    V2 improvements:
    1. Uses position-aware n-grams for all tokens
    2. Uses element-only context for element prediction
    3. Leverages chemical family hints for element prediction

    Uses structural rules to constrain valid tokens, then n-gram probabilities
    to rank within valid set. Falls back to structural-only if n-gram has no
    data for the context.
    """

    def __init__(
        self,
        ngram: Optional[NGramDraft] = None,
        structural: Optional[StructuralDraft] = None,
        ngram_weight: float = 0.7,
    ):
        """
        Args:
            ngram: Trained n-gram model
            structural: Trained structural model
            ngram_weight: Weight for n-gram predictions (0-1)
        """
        self.ngram = ngram or NGramDraft()
        self.structural = structural or StructuralDraft()
        self.ngram_weight = ngram_weight

        # Build element indices set for fast lookup
        self._element_indices: Set[int] = set()
        for elem in ELEMENTS[1:]:  # Skip empty element
            if elem in TOKEN_TO_IDX:
                self._element_indices.add(TOKEN_TO_IDX[elem])

    def train(self, formulas: List[str], max_len: int = 60):
        """
        Train both component models on formula data (V2).

        Args:
            formulas: List of formula strings
            max_len: Maximum sequence length
        """
        print("=" * 60)
        print("Training V2 Hybrid Draft Model")
        print("=" * 60)
        self.ngram.train(formulas, max_len=max_len)
        self.structural.train(formulas)
        print("=" * 60)
        print("V2 Hybrid Draft Model training complete")
        print("=" * 60)

    def predict_next(self, tokens: List[int], position: Optional[int] = None) -> Optional[int]:
        """
        Predict next token using hybrid approach (V2: position-aware + element-context).

        V2 strategy:
        1. Get valid tokens from structural model
        2. Get position-aware n-gram candidates (works for ALL token types)
        3. If top n-gram candidate is valid, use it
        4. If predicting an element AND n-gram didn't help:
           a. Try element-only context prediction (skip digits)
           b. Fall back to position-aware element unigram
        5. Fall back to structural if no match

        Args:
            tokens: Current token sequence
            position: Current sequence position (optional)

        Returns:
            Predicted next token index
        """
        if position is None:
            position = len(tokens)

        # Get structurally valid tokens
        state, info = self.structural.parse_state(tokens)
        valid_tokens = self.structural.get_valid_next_tokens(state, info)

        if not valid_tokens:
            return END_IDX

        # V2: FIRST try position-aware n-gram (handles structural tokens well)
        ngram_candidates = self.ngram.get_top_k(tokens, k=10, position=position)

        # Find best candidate that's structurally valid
        for idx, prob in ngram_candidates:
            if idx in valid_tokens:
                return idx

        # V2: If n-gram didn't find anything valid, try element-specific prediction
        # This only applies when elements are valid AND we're in a context where
        # the next token MUST be an element (START or CLOSE_PAREN states only)
        valid_elements = valid_tokens & self._element_indices
        is_element_only_context = (
            state in [FormulaState.START, FormulaState.CLOSE_PAREN] and
            len(valid_elements) > 0
        )

        if is_element_only_context:
            # V2: Use element-only context for element prediction
            elem_position = info.get('element_position', 0)

            # Try element-specific prediction from n-gram (element-only context)
            elem_pred = self.ngram.predict_next_element(
                tokens, elem_position, valid_elements
            )
            if elem_pred is not None:
                return elem_pred

            # Try element-specific top-k
            elem_candidates = self.ngram.get_top_k_elements(tokens, k=10, elem_position=elem_position)
            for idx, prob in elem_candidates:
                if idx in valid_elements:
                    return idx

        # Fall back to structural prediction
        return self.structural.predict_next(tokens)

    def draft_k_tokens(self, tokens: List[int], k: int = 5) -> List[int]:
        """
        Draft k tokens using hybrid predictions (V2: position-aware).

        Args:
            tokens: Current token sequence
            k: Number of tokens to draft

        Returns:
            List of k drafted token indices
        """
        context = list(tokens)
        drafted = []
        position = len(tokens)

        for _ in range(k):
            next_token = self.predict_next(context, position=position)
            if next_token is None or next_token == END_IDX:
                drafted.append(END_IDX)
                break
            drafted.append(next_token)
            context.append(next_token)
            position += 1

        # Pad with END if needed
        while len(drafted) < k:
            drafted.append(END_IDX)

        return drafted[:k]

    def draft_k_tokens_batch(
        self,
        tokens_batch: torch.Tensor,
        k: int = 5,
    ) -> torch.Tensor:
        """
        Draft k tokens for a batch of sequences.

        Args:
            tokens_batch: [batch, seq_len] tensor of current tokens
            k: Number of tokens to draft

        Returns:
            [batch, k] tensor of drafted tokens
        """
        batch_size = tokens_batch.shape[0]
        device = tokens_batch.device

        drafted = []
        for i in range(batch_size):
            # Convert to list, removing PAD tokens
            seq = tokens_batch[i].tolist()
            seq = [t for t in seq if t != PAD_IDX]

            # Draft k tokens
            draft = self.draft_k_tokens(seq, k=k)
            drafted.append(draft)

        return torch.tensor(drafted, device=device, dtype=torch.long)

    def save(self, path: Path):
        """Save hybrid model (saves both components)."""
        path = Path(path)
        ngram_path = path.with_suffix('.ngram.pkl')
        struct_path = path.with_suffix('.struct.pkl')

        self.ngram.save(ngram_path)
        self.structural.save(struct_path)

        # Save metadata
        meta = {
            'ngram_weight': self.ngram_weight,
            'ngram_path': str(ngram_path),
            'structural_path': str(struct_path),
        }
        with open(path, 'wb') as f:
            pickle.dump(meta, f)

        print(f"Saved hybrid model to {path}")

    @classmethod
    def load(cls, path: Path) -> 'HybridDraft':
        """Load hybrid model from saved files."""
        path = Path(path)

        with open(path, 'rb') as f:
            meta = pickle.load(f)

        ngram_path = Path(meta.get('ngram_path', path.with_suffix('.ngram.pkl')))
        struct_path = Path(meta.get('structural_path', path.with_suffix('.struct.pkl')))

        ngram = NGramDraft.load(ngram_path)
        structural = StructuralDraft.load(struct_path)

        model = cls(
            ngram=ngram,
            structural=structural,
            ngram_weight=meta.get('ngram_weight', 0.7),
        )

        print(f"Loaded hybrid model from {path}")
        return model


# =============================================================================
# Speculative Decoding Utilities
# =============================================================================

def verify_draft_tokens(
    model_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    temperature: float = 0.8,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Verify drafted tokens against main model logits.

    Uses rejection sampling: accept draft token if it's likely enough under
    the main model distribution.

    Args:
        model_logits: [batch, k, vocab_size] logits from main model
        draft_tokens: [batch, k] drafted token indices
        temperature: Sampling temperature
        threshold: Minimum probability threshold (0 = always sample)

    Returns:
        accepted_tokens: [batch, n_accepted] tensor of accepted tokens
        log_probs: [batch, n_accepted] log probabilities
        n_accepted: Number of accepted tokens (same for all batch)
    """
    batch_size, k, vocab_size = model_logits.shape
    device = model_logits.device

    # Apply temperature
    scaled_logits = model_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs_all = F.log_softmax(scaled_logits, dim=-1)

    # Check each draft position
    accepted_mask = torch.ones(batch_size, k, dtype=torch.bool, device=device)

    for pos in range(k):
        # Get probability of drafted token at this position
        draft_token = draft_tokens[:, pos]  # [batch]
        draft_prob = probs[:, pos, :].gather(1, draft_token.unsqueeze(1)).squeeze(1)  # [batch]

        # Sample uniform random for rejection sampling
        r = torch.rand(batch_size, device=device)

        # Accept if r < draft_prob (standard rejection sampling)
        # With threshold, accept if draft_prob > threshold OR r < draft_prob/threshold
        if threshold > 0:
            accept = (draft_prob > threshold) | (r < draft_prob)
        else:
            accept = r < draft_prob

        # Once we reject, all subsequent positions are also rejected
        if pos > 0:
            accept = accept & accepted_mask[:, pos - 1]

        accepted_mask[:, pos] = accept

    # Find first rejection position for each sequence
    # (number of accepted tokens = position of first False)
    first_reject = torch.argmin(accepted_mask.int(), dim=1)  # [batch]

    # If all accepted, first_reject will be 0 (argmin of all 1s)
    # We need to handle this case
    all_accepted = accepted_mask.all(dim=1)
    first_reject = torch.where(all_accepted, torch.tensor(k, device=device), first_reject)

    # Use minimum across batch for uniform output shape
    n_accepted = first_reject.min().item()

    if n_accepted > 0:
        accepted_tokens = draft_tokens[:, :n_accepted]
        log_probs = log_probs_all[:, :n_accepted, :].gather(
            2, accepted_tokens.unsqueeze(-1)
        ).squeeze(-1)
    else:
        accepted_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        log_probs = torch.empty(batch_size, 0, device=device)

    return accepted_tokens, log_probs, n_accepted


def sample_from_adjusted_distribution(
    model_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    n_accepted: int,
    temperature: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample the next token after rejection using adjusted distribution.

    When a draft token is rejected at position n_accepted, we sample from:
    p_adjusted(x) = max(0, p_model(x) - p_draft(x)) / Z

    For simplicity with n-gram draft (discrete), we just sample from model.

    Args:
        model_logits: [batch, k, vocab_size] logits from main model
        draft_tokens: [batch, k] drafted tokens
        n_accepted: Number of accepted tokens
        temperature: Sampling temperature

    Returns:
        next_token: [batch, 1] sampled token
        log_prob: [batch, 1] log probability
    """
    batch_size = model_logits.shape[0]
    device = model_logits.device

    # Get logits at the rejection position
    if n_accepted < model_logits.shape[1]:
        logits = model_logits[:, n_accepted, :]  # [batch, vocab_size]
    else:
        # All tokens accepted, need one more step
        logits = model_logits[:, -1, :]

    # Apply temperature and sample
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)

    # Sample from distribution
    next_token = torch.multinomial(probs, num_samples=1)
    next_log_prob = log_probs.gather(1, next_token)

    return next_token, next_log_prob


# =============================================================================
# Factory Functions
# =============================================================================

def build_draft_model(
    formulas: List[str],
    max_len: int = 60,
    save_path: Optional[Path] = None,
) -> HybridDraft:
    """
    Build and optionally save a hybrid draft model from training data.

    Args:
        formulas: List of formula strings
        max_len: Maximum sequence length
        save_path: Optional path to save model

    Returns:
        Trained HybridDraft model
    """
    model = HybridDraft()
    model.train(formulas, max_len=max_len)

    if save_path:
        model.save(save_path)

    return model


def load_or_build_draft_model(
    formulas: List[str],
    cache_path: Path,
    max_len: int = 60,
    force_rebuild: bool = False,
) -> HybridDraft:
    """
    Load draft model from cache or build if not found.

    Args:
        formulas: List of formula strings (for building)
        cache_path: Path to cache file
        max_len: Maximum sequence length
        force_rebuild: Force rebuild even if cache exists

    Returns:
        HybridDraft model
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        try:
            return HybridDraft.load(cache_path)
        except Exception as e:
            print(f"Failed to load draft model cache: {e}")

    # Build and save
    model = build_draft_model(formulas, max_len=max_len, save_path=cache_path)
    return model
