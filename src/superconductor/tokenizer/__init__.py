"""
Fraction-aware tokenizer for V13.0 semantic fraction tokenization.

Replaces character-level fraction tokenization (e.g., `(`, `1`, `7`, `/`, `2`, `0`, `)`)
with single semantic fraction tokens (e.g., `FRAC:17/20`), eliminating cascading digit
errors and reducing sequence length by ~40-60%.
"""

from .fraction_tokenizer import FractionAwareTokenizer

__all__ = ['FractionAwareTokenizer']
