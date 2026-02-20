#!/usr/bin/env python3
"""
Build isotope vocabulary JSON from the existing ISOTOPE_DATABASE.

Generates data/isotope_vocab.json with 291 isotope tokens sorted by
(atomic_number, mass_number) for deterministic ordering.

Usage:
    python scripts/build_isotope_vocab.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from superconductor.encoders.isotope_properties import ISOTOPE_DATABASE
from superconductor.tokenizer.fraction_tokenizer import ELEMENTS


def build_isotope_vocab(output_path: str):
    """Generate isotope vocabulary JSON from ISOTOPE_DATABASE.

    Sorting: by atomic number (element order in periodic table), then mass number.
    Token format: "{mass}{symbol}" e.g. "18O", "2H", "63Cu"
    """
    isotopes = []
    element_isotope_counts = {}

    # Iterate elements in periodic table order (ELEMENTS[1:] = H, He, Li, ...)
    for elem in ELEMENTS[1:]:
        if elem not in ISOTOPE_DATABASE:
            continue
        mass_numbers = sorted(ISOTOPE_DATABASE[elem].keys())
        element_isotope_counts[elem] = len(mass_numbers)
        for mass_num in mass_numbers:
            isotopes.append(f"{mass_num}{elem}")

    vocab = {
        "version": "V14.0",
        "description": "Isotope vocabulary for V14.0 tokenizer — single semantic tokens per isotope",
        "source": "src/superconductor/encoders/isotope_properties.py (ISOTOPE_DATABASE)",
        "n_isotopes": len(isotopes),
        "n_elements_with_isotopes": len(element_isotope_counts),
        "isotopes": isotopes,
        "element_isotope_counts": element_isotope_counts,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"Isotope vocabulary saved to {output_path}")
    print(f"  {len(isotopes)} isotopes across {len(element_isotope_counts)} elements")
    print(f"  First 10: {isotopes[:10]}")
    print(f"  Last 10: {isotopes[-10:]}")
    return vocab


if __name__ == '__main__':
    output = str(PROJECT_ROOT / 'data' / 'isotope_vocab.json')
    build_isotope_vocab(output)
