# Fraction Tokenization Refactor — Implementation Plan for Claude Code
## Superconductor VAE: Character-Level Fractions → Atomic Fraction Tokens

**Target model**: V12.41 → V13.0  
**Scope**: Tokenizer vocabulary, data preprocessing pipeline, decoder architecture, checkpoint transfer  
**Goal**: Replace character-by-character fraction encoding with single semantic fraction tokens

---

## Background and Motivation

**Current behavior** (character-level):
```
Ba → 2 → Ca → 2 → Cu → 3 → O → ( → 1 → 7 → / → 2 → 0 → )
```
Each fraction like `(17/20)` requires 7 autoregressive steps. Each step is an opportunity for error (wrong digit, wrong denominator, mismatched GCD).

**Proposed behavior** (semantic token):
```
Ba → 2 → Ca → 2 → Cu → 3 → O → [FRAC:17/20]
```
The fraction is one decoding step. The seq_len↔error correlation (~0.5) in training logs means this directly reduces error rate proportional to sequence compression.

**Why this is the right fix** rather than adding more decoder capacity: The model isn't failing because it lacks expressiveness — it's failing because it must coordinate 5-7 interdependent character predictions to emit one number. A single cross-entropy decision over a closed fraction vocabulary is structurally easier and eliminates the denominator-doubling artifact (A2 in constraint zoo) at the architecture level rather than via a penalty.

---

## Phase 0: Vocabulary Audit (Run First, Before Any Code Changes)

This phase determines the fraction vocabulary size and informs all subsequent decisions. **Do not skip this.**

### Step 0.1 — Extract All Fractions from Training Data

```python
# scripts/audit_fractions.py
import re
import json
from collections import Counter
from pathlib import Path

def extract_fractions(formula_string):
    """Extract all (p/q) patterns from a formula string."""
    return re.findall(r'\((\d+)/(\d+)\)', formula_string)

def audit_dataset(data_path):
    """
    Load training data and extract full fraction statistics.
    Adjust data loading to match actual file format.
    """
    formulas = []
    # Load from whatever format the training data is in
    # (CSV, JSON, pickle — adjust accordingly)
    with open(data_path) as f:
        data = json.load(f)
    formulas = [entry['formula'] for entry in data]
    
    fraction_counter = Counter()
    numerator_counter = Counter()
    denominator_counter = Counter()
    formula_fraction_counts = []
    
    for formula in formulas:
        fracs = extract_fractions(formula)
        formula_fraction_counts.append(len(fracs))
        for num, den in fracs:
            fraction_str = f'{num}/{den}'
            fraction_counter[fraction_str] += 1
            numerator_counter[int(num)] += 1
            denominator_counter[int(den)] += 1
    
    return {
        'unique_fractions': len(fraction_counter),
        'total_fraction_occurrences': sum(fraction_counter.values()),
        'top_50_fractions': fraction_counter.most_common(50),
        'numerator_distribution': sorted(numerator_counter.items()),
        'denominator_distribution': sorted(denominator_counter.items()),
        'max_numerator': max(numerator_counter.keys()),
        'max_denominator': max(denominator_counter.keys()),
        'fractions_per_formula_mean': sum(formula_fraction_counts) / len(formula_fraction_counts),
        'fractions_per_formula_max': max(formula_fraction_counts),
        'coverage_by_top_n': {  # What % of occurrences covered by top-N fraction types?
            100: sum(v for _, v in fraction_counter.most_common(100)) / sum(fraction_counter.values()),
            500: sum(v for _, v in fraction_counter.most_common(500)) / sum(fraction_counter.values()),
            1000: sum(v for _, v in fraction_counter.most_common(1000)) / sum(fraction_counter.values()),
            'all': 1.0
        },
        'full_counter': dict(fraction_counter)
    }

if __name__ == '__main__':
    results = audit_dataset('data/training_data.json')  # adjust path
    print(f"Unique fraction types: {results['unique_fractions']}")
    print(f"Max numerator: {results['max_numerator']}")
    print(f"Max denominator: {results['max_denominator']}")
    print(f"Mean fractions per formula: {results['fractions_per_formula_mean']:.2f}")
    print(f"\nTop 20 fractions:")
    for frac, count in results['top_50_fractions'][:20]:
        print(f"  ({frac}): {count}")
    print(f"\nCoverage by top-N:")
    for n, cov in results['coverage_by_top_n'].items():
        print(f"  Top {n}: {cov:.3%}")
    
    # Save full results for vocabulary construction decision
    with open('audit_results.json', 'w') as f:
        json.dump(results, f, indent=2)
```

### Step 0.2 — Decision Gate Based on Audit Results

After running the audit, answer these questions before proceeding:

**Q1: How many unique fraction types exist?**
- < 500: Use full closed vocabulary (all fractions as tokens). Simplest approach.
- 500–2000: Use top-N by frequency with a `[FRAC_UNK]` fallback token. Recommend N such that coverage ≥ 99.5%.
- > 2000: Consider fraction *decomposition tokens* instead (see Appendix A). Unlikely given SuperCon data constraints.

**Q2: Are denominators bounded?**
- If max denominator < 200: Closed vocabulary is clean.
- If there are sparse very-large-denominator fractions: These are almost certainly GCD artifacts (addressed by A2). Canonicalize first (Step 1.1), then re-audit. Expect denominator distribution to collapse significantly.

**Q3: What is the sequence length reduction?**
- Each fraction currently costs: 2 (parentheses) + len(num_digits) + 1 (slash) + len(den_digits) steps
- For (17/20): 7 steps → 1 step. Reduction = 6 steps per fraction.
- Mean fractions per formula × 6 = expected mean sequence length reduction
- Record this — it's your theoretical error rate improvement estimate.

**Expected audit results** (based on known SuperCon characteristics):
- Unique fractions: ~300–800 (SuperCon has structured chemistry, not arbitrary rationals)
- Most common: (1/2), (1/4), (3/4), (1/10), (3/10), (1/5), (2/5), (3/5), (1/20), (17/20) — simple doping fractions
- Top 100 fractions cover ~95%+ of occurrences
- Max denominator: likely < 1000, with most < 100

---

## Phase 1: Preprocessing Pipeline

Run preprocessing on the full training/validation/test datasets **before** any model code changes. The output of this phase is canonicalized, fraction-tokenized data files that can be used directly.

### Step 1.1 — Canonicalize Existing Fractions (A2 Fix)

Do this before building the vocabulary, because non-canonical fractions (`6/200` → `3/100`) inflate the vocabulary count artificially.

```python
# scripts/canonicalize_fractions.py
import re
from math import gcd
from pathlib import Path

def canonicalize_fraction(num_str, den_str):
    """Reduce fraction to canonical form."""
    num, den = int(num_str), int(den_str)
    g = gcd(num, den)
    return num // g, den // g

def canonicalize_formula(formula_string):
    """Replace all (p/q) in formula with canonical (p'/q')."""
    def replace_fraction(match):
        num, den = canonicalize_fraction(match.group(1), match.group(2))
        return f'({num}/{den})'
    return re.sub(r'\((\d+)/(\d+)\)', replace_fraction, formula_string)

def canonicalize_dataset(input_path, output_path):
    """Canonicalize all fractions in dataset. Log changes for audit."""
    import json
    with open(input_path) as f:
        data = json.load(f)
    
    changes = 0
    for entry in data:
        original = entry['formula']
        canonical = canonicalize_formula(original)
        if canonical != original:
            changes += 1
            entry['formula'] = canonical
            entry['formula_original'] = original  # Keep for debugging
    
    print(f"Canonicalized {changes}/{len(data)} formulas ({changes/len(data):.1%})")
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return data

# Run on all splits
for split in ['train', 'val', 'test']:
    canonicalize_dataset(f'data/{split}.json', f'data/{split}_canonical.json')
```

### Step 1.2 — Build Fraction Vocabulary

After canonicalization, build the definitive fraction vocabulary.

```python
# scripts/build_fraction_vocab.py
import json
from collections import Counter
import re

def build_fraction_vocabulary(
    canonical_data_path,
    coverage_threshold=0.999,  # Include fractions until 99.9% coverage
    min_count=1,               # Include all fractions seen at least once
    output_path='data/fraction_vocab.json'
):
    with open(canonical_data_path) as f:
        data = json.load(f)
    
    counter = Counter()
    for entry in data:
        fracs = re.findall(r'\((\d+)/(\d+)\)', entry['formula'])
        for num, den in fracs:
            counter[f'{num}/{den}'] += 1
    
    total = sum(counter.values())
    
    # Sort by frequency (most common first for stable indexing)
    sorted_fracs = counter.most_common()
    
    # Build vocab with coverage threshold
    vocab = []
    cumulative = 0
    for frac, count in sorted_fracs:
        if count < min_count:
            break
        vocab.append(frac)
        cumulative += count
        if cumulative / total >= coverage_threshold:
            break
    
    # Always include special fractions that matter to physics
    # (even if rare in training data — they appear in holdout)
    ALWAYS_INCLUDE = [
        '1/2', '1/3', '2/3', '1/4', '3/4', '1/5', '2/5', '3/5', '4/5',
        '1/10', '3/10', '7/10', '9/10', '1/20', '3/20', '17/20', '19/20',
        '1/100', '3/100', '5/100', '17/100', '83/100',
        '1/25', '4/25', '21/25',  # appear in holdout (U, V dopants)
        '71/100', '29/100', '41/250', '57/1000',  # A15 holdout fractions
    ]
    for f in ALWAYS_INCLUDE:
        if f not in vocab:
            vocab.append(f)
    
    vocab_dict = {
        'fractions': vocab,
        'fraction_to_id': {f: i for i, f in enumerate(vocab)},
        'total_unique': len(counter),
        'vocab_size': len(vocab),
        'coverage': sum(counter[f] for f in vocab if f in counter) / total,
        'has_unk': True,  # Always include UNK for generation safety
    }
    
    with open(output_path, 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    
    print(f"Fraction vocabulary: {len(vocab)} tokens")
    print(f"Coverage: {vocab_dict['coverage']:.4%}")
    print(f"UNK rate in training: {1 - vocab_dict['coverage']:.4%}")
    
    return vocab_dict
```

### Step 1.3 — Convert Dataset to Fraction-Token Format

The key question is: what intermediate representation do you use for training? Two options:

**Option A — Fraction tokens in the same sequence (recommended)**  
Keep one unified token sequence. Fraction tokens are just additional vocabulary entries alongside element symbols and integer coefficients.

```
['Ba', '2', 'Ca', '2', 'Cu', '3', 'O', 'FRAC:17/20']
# vs current:
['Ba', '2', 'Ca', '2', 'Cu', '3', 'O', '(', '1', '7', '/', '2', '0', ')']
```

**Option B — Separate fraction prediction head**  
Keep element/integer sequence as before but replace the `( → digits → / → digits → )` subsequence with a single `[FRAC]` marker token, and use a separate cross-entropy head over fraction vocab to predict which fraction.

Option A is simpler and fully unified. Option B allows more targeted fine-tuning but adds architectural complexity. **Proceed with Option A** unless Option B is specifically requested.

```python
# scripts/convert_to_fraction_tokens.py
import json
import re

def formula_to_token_sequence(formula_string, fraction_vocab):
    """
    Convert formula string to list of tokens.
    Fractions become single 'FRAC:p/q' tokens.
    Elements, integers stay as string tokens.
    """
    tokens = []
    # Tokenization regex: matches elements, integers, fractions, brackets
    pattern = r'([A-Z][a-z]?|\d+|\(\d+/\d+\)|[()])'
    
    i = 0
    while i < len(formula_string):
        # Try to match fraction first (greedy)
        frac_match = re.match(r'\((\d+)/(\d+)\)', formula_string[i:])
        if frac_match:
            num, den = frac_match.group(1), frac_match.group(2)
            frac_key = f'{num}/{den}'
            if frac_key in fraction_vocab['fraction_to_id']:
                tokens.append(f'FRAC:{frac_key}')
            else:
                tokens.append('FRAC:UNK')  # Should be rare post-canonicalization
            i += len(frac_match.group(0))
            continue
        
        # Try element symbol
        elem_match = re.match(r'[A-Z][a-z]?', formula_string[i:])
        if elem_match:
            tokens.append(elem_match.group(0))
            i += len(elem_match.group(0))
            continue
        
        # Try integer
        int_match = re.match(r'\d+', formula_string[i:])
        if int_match:
            tokens.append(int_match.group(0))
            i += len(int_match.group(0))
            continue
        
        # Skip other characters (spaces, etc.)
        i += 1
    
    return tokens

def convert_dataset(data_path, fraction_vocab_path, output_path):
    with open(data_path) as f:
        data = json.load(f)
    with open(fraction_vocab_path) as f:
        fraction_vocab = json.load(f)
    
    seq_lengths_before = []
    seq_lengths_after = []
    
    for entry in data:
        # Count current character-level length for comparison
        formula = entry['formula']
        # Rough character count of current tokenization
        seq_lengths_before.append(len(re.sub(r'\s', '', formula)))
        
        tokens = formula_to_token_sequence(formula, fraction_vocab)
        entry['tokens'] = tokens
        seq_lengths_after.append(len(tokens))
    
    print(f"Mean sequence length: {sum(seq_lengths_before)/len(seq_lengths_before):.1f} → "
          f"{sum(seq_lengths_after)/len(seq_lengths_after):.1f}")
    print(f"Max sequence length: {max(seq_lengths_before)} → {max(seq_lengths_after)}")
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
```

---

## Phase 2: Tokenizer Class Refactor

The tokenizer is the central object that maps between formula strings and integer indices. This needs a clean rewrite (not a patch) to support the new vocabulary.

```python
# tokenizer/fraction_tokenizer.py

import re
import json
from math import gcd
from typing import List, Optional, Dict

class FractionAwareTokenizer:
    """
    Unified tokenizer for superconductor formulas with atomic fraction tokens.
    
    Vocabulary structure:
        [SPECIAL TOKENS]     : PAD, BOS, EOS, UNK, FRAC_UNK
        [ELEMENT TOKENS]     : H, He, Li, Be, B, ... (all 118 elements)
        [INTEGER TOKENS]     : 1, 2, 3, ... MAX_INT
        [FRACTION TOKENS]    : FRAC:1/2, FRAC:1/4, FRAC:3/4, ...
    """
    
    # Special token definitions
    PAD_TOKEN = '[PAD]'
    BOS_TOKEN = '[BOS]'
    EOS_TOKEN = '[EOS]'
    UNK_TOKEN = '[UNK]'
    FRAC_UNK_TOKEN = '[FRAC_UNK]'
    
    ELEMENTS = [
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
    
    def __init__(
        self,
        fraction_vocab_path: str,
        max_integer: int = 20,  # Max stoichiometric integer (e.g., O18, Cu3)
    ):
        with open(fraction_vocab_path) as f:
            frac_data = json.load(f)
        
        self.fraction_list = frac_data['fractions']  # ordered list
        self.max_integer = max_integer
        
        # Build unified vocabulary
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Construct token→id and id→token mappings."""
        vocab = []
        
        # Special tokens (fixed indices 0-4)
        vocab += [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN,
                  self.UNK_TOKEN, self.FRAC_UNK_TOKEN]
        
        # Element tokens
        vocab += self.ELEMENTS
        
        # Integer tokens (1 through max_integer)
        vocab += [str(i) for i in range(1, self.max_integer + 1)]
        
        # Fraction tokens
        vocab += [f'FRAC:{f}' for f in self.fraction_list]
        
        self.vocab = vocab
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(vocab)}
        
        # Convenience: record index ranges for each token type
        self.special_range = (0, 5)
        self.element_range = (5, 5 + len(self.ELEMENTS))
        self.integer_range = (5 + len(self.ELEMENTS), 
                              5 + len(self.ELEMENTS) + self.max_integer)
        self.fraction_range = (5 + len(self.ELEMENTS) + self.max_integer,
                               len(vocab))
        
        # Special token IDs for quick access
        self.PAD_ID = self.token_to_id[self.PAD_TOKEN]
        self.BOS_ID = self.token_to_id[self.BOS_TOKEN]
        self.EOS_ID = self.token_to_id[self.EOS_TOKEN]
        self.UNK_ID = self.token_to_id[self.UNK_TOKEN]
        self.FRAC_UNK_ID = self.token_to_id[self.FRAC_UNK_TOKEN]
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    @property
    def n_fraction_tokens(self) -> int:
        lo, hi = self.fraction_range
        return hi - lo
    
    def _canonicalize(self, num: int, den: int):
        g = gcd(num, den)
        return num // g, den // g
    
    def encode(self, formula_string: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert formula string to token ID sequence.
        
        Args:
            formula_string: e.g. "Ba2Ca2Cu3O(17/20)"
            add_special_tokens: prepend BOS, append EOS
        
        Returns:
            List of integer token IDs
        """
        ids = []
        if add_special_tokens:
            ids.append(self.BOS_ID)
        
        i = 0
        while i < len(formula_string):
            # Fraction: highest priority match
            frac_match = re.match(r'\((\d+)/(\d+)\)', formula_string[i:])
            if frac_match:
                num, den = self._canonicalize(int(frac_match.group(1)), 
                                              int(frac_match.group(2)))
                frac_token = f'FRAC:{num}/{den}'
                ids.append(self.token_to_id.get(frac_token, self.FRAC_UNK_ID))
                i += len(frac_match.group(0))
                continue
            
            # Element (two-char first, then one-char)
            two_char = formula_string[i:i+2]
            if two_char in self.token_to_id:
                ids.append(self.token_to_id[two_char])
                i += 2
                continue
            
            one_char = formula_string[i:i+1]
            if one_char in self.token_to_id:
                ids.append(self.token_to_id[one_char])
                i += 1
                continue
            
            # Integer
            int_match = re.match(r'\d+', formula_string[i:])
            if int_match:
                val = int(int_match.group(0))
                tok = str(val) if val <= self.max_integer else self.UNK_TOKEN
                ids.append(self.token_to_id.get(tok, self.UNK_ID))
                i += len(int_match.group(0))
                continue
            
            # Skip unknown characters (don't crash)
            ids.append(self.UNK_ID)
            i += 1
        
        if add_special_tokens:
            ids.append(self.EOS_ID)
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Convert token ID sequence back to formula string."""
        parts = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, self.UNK_TOKEN)
            if skip_special and tok in {self.PAD_TOKEN, self.BOS_TOKEN, 
                                         self.EOS_TOKEN, self.UNK_TOKEN}:
                if tok == self.EOS_TOKEN:
                    break  # Stop at EOS
                continue
            
            if tok.startswith('FRAC:'):
                frac = tok[5:]  # strip 'FRAC:'
                parts.append(f'({frac})')
            elif tok == self.FRAC_UNK_TOKEN:
                parts.append('(?/?)')  # placeholder for unknown fractions
            else:
                parts.append(tok)
        
        return ''.join(parts)
    
    def is_fraction_token(self, token_id: int) -> bool:
        lo, hi = self.fraction_range
        return lo <= token_id < hi or token_id == self.FRAC_UNK_ID
    
    def fraction_token_to_value(self, token_id: int) -> Optional[float]:
        """Convert a fraction token ID to its float value."""
        tok = self.id_to_token.get(token_id, '')
        if tok.startswith('FRAC:'):
            num, den = tok[5:].split('/')
            return int(num) / int(den)
        return None
    
    def save(self, path: str):
        """Save tokenizer state for exact reconstruction."""
        import json
        state = {
            'vocab': self.vocab,
            'fraction_list': self.fraction_list,
            'max_integer': self.max_integer,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from saved state."""
        with open(path) as f:
            state = json.load(f)
        # Reconstruct — create minimal fraction_vocab_path stub
        import tempfile
        vocab_data = {'fractions': state['fraction_list']}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(vocab_data, f)
            tmp_path = f.name
        tokenizer = cls(tmp_path, max_integer=state['max_integer'])
        return tokenizer


def test_tokenizer(tokenizer):
    """Smoke test to verify round-trip correctness."""
    test_cases = [
        "Y1Ba2Cu3O(17/20)",
        "Bi(9/5)Pb(1/5)Sr2Ca2Cu3O10",
        "Mg1B(17/20)C(3/20)",
        "Hg(17/20)Re(3/20)Ba(83/50)Sr(17/50)Ca2Cu3O8",  # Hg mode-collapse case
        "Nb(79/100)Al(71/100)Ge(29/100)",  # A15 holdout
    ]
    
    print("Tokenizer round-trip tests:")
    for formula in test_cases:
        ids = tokenizer.encode(formula)
        decoded = tokenizer.decode(ids)
        match = decoded == formula
        seq_len = len(ids)
        print(f"  {'✓' if match else '✗'} [{seq_len} tokens] {formula}")
        if not match:
            print(f"    → Got: {decoded}")
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Fraction tokens: {tokenizer.n_fraction_tokens}")
    print(f"Element range: {tokenizer.element_range}")
    print(f"Fraction range: {tokenizer.fraction_range}")
```

---

## Phase 3: Model Architecture Changes

### Step 3.1 — Embedding Layer

The embedding layer grows to accommodate the new fraction vocabulary. Everything else in the encoder stays identical.

```python
# In your model definition, update the token embedding:

class FractionAwareEmbedding(nn.Module):
    """
    Drop-in replacement for the existing token embedding.
    Supports initialization from a pre-trained embedding table (weight transfer).
    """
    def __init__(self, tokenizer: FractionAwareTokenizer, embed_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=tokenizer.PAD_ID
        )
    
    def forward(self, token_ids):
        return self.embedding(token_ids)
    
    def transfer_weights_from_v12(self, old_embedding: nn.Embedding, old_tokenizer):
        """
        Copy weights for tokens that exist in both old and new vocabularies.
        New fraction tokens are initialized randomly (they need to be learned).
        
        Args:
            old_embedding: The V12.41 embedding table
            old_tokenizer: The V12.41 tokenizer (for token→id mapping)
        """
        with torch.no_grad():
            transferred = 0
            for token, new_id in self.tokenizer.token_to_id.items():
                if token in old_tokenizer.token_to_id:
                    old_id = old_tokenizer.token_to_id[token]
                    self.embedding.weight[new_id] = old_embedding.weight[old_id]
                    transferred += 1
            
            print(f"Weight transfer: {transferred}/{self.tokenizer.vocab_size} tokens "
                  f"({transferred/self.tokenizer.vocab_size:.1%})")
            print(f"New (random init): {self.tokenizer.vocab_size - transferred} tokens")
        # Fraction token embeddings remain randomly initialized — they will be learned
        # from the character-level meaning encoded in the now-reused encoder context
```

### Step 3.2 — Decoder Output Head

The output head (the linear layer that maps hidden state → logits → next token) also needs updating. This is straightforward: it just maps to the new vocab size.

```python
# In decoder:
# OLD:
self.output_projection = nn.Linear(decoder_hidden_dim, old_vocab_size)

# NEW:
self.output_projection = nn.Linear(decoder_hidden_dim, tokenizer.vocab_size)

# Weight transfer for output projection (same principle as embedding):
def transfer_output_projection(new_proj, old_proj, new_tokenizer, old_tokenizer):
    with torch.no_grad():
        for token, new_id in new_tokenizer.token_to_id.items():
            if token in old_tokenizer.token_to_id:
                old_id = old_tokenizer.token_to_id[token]
                new_proj.weight[new_id] = old_proj.weight[old_id]
                new_proj.bias[new_id] = old_proj.bias[old_id]
```

### Step 3.3 — Remove or Repurpose the numden_head

**Current state**: The `numden_head` is a regression head that predicts `(numerator, denominator)` as continuous values via `log1p` transform. This was needed because fractions were not direct vocabulary tokens.

**After this change**: The decoder itself predicts which fraction token to emit via cross-entropy, so the fraction value is implicit in the token choice. The `numden_head` is no longer needed for generation.

**Options**:

*Option A — Remove numden_head entirely*  
Clean. The fraction prediction accuracy is now measured as token classification accuracy on fraction tokens, not via regression MAE.

*Option B — Keep numden_head as auxiliary consistency loss*  
After emitting a `FRAC:17/20` token, compute `numden_head(hidden_state)` and check it predicts `17/20`. This adds a soft consistency signal during training. Minimal overhead. Recommended if the head was contributing meaningfully to Tc prediction.

*Option C — Repurpose numden_head as Tc-conditioned fraction prior*  
The fraction value (doping level) correlates strongly with Tc. Keep a lightweight regression head that predicts expected fraction value conditioned on the latent Z and family, then use this to bias the fraction token logits during generation. This could improve generation quality by making the model aware that "for this Tc, the doping level should be ~0.15, so (3/20) is more likely than (1/2)."

**Recommendation**: Option A for the initial V13.0 implementation. If fraction generation quality is still imperfect after training, add Option C as V13.1.

```python
# Option A — clean removal:
# Delete the numden_head module and all references to it in the training loop.
# Remove numden_loss from the composite training loss.

# Option B — auxiliary consistency loss:
class NumdenConsistencyLoss(nn.Module):
    """Auxiliary loss: hidden state should predict the fraction value it just emitted."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # predicts (log1p(num), log1p(den))
        )
    
    def forward(self, hidden_states, token_ids, tokenizer):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            token_ids: [batch, seq_len] — the tokens being predicted
            tokenizer: for looking up fraction values
        """
        # Only compute loss at positions where target is a fraction token
        is_fraction = torch.tensor([
            [tokenizer.is_fraction_token(t.item()) for t in seq]
            for seq in token_ids
        ], dtype=torch.bool)
        
        if not is_fraction.any():
            return torch.tensor(0.0)
        
        frac_hidden = hidden_states[is_fraction]  # [n_fracs, hidden_dim]
        frac_preds = self.head(frac_hidden)        # [n_fracs, 2]
        
        frac_targets = torch.stack([
            torch.tensor([
                math.log1p(num), math.log1p(den)
            ])
            for tid in token_ids[is_fraction]
            for num, den in [tokenizer.id_to_token[tid.item()][5:].split('/')]
        ])
        
        return F.mse_loss(frac_preds, frac_targets)
```

### Step 3.4 — GCD Canonicality Loss (A2) Becomes Free

With atomic fraction tokens, the A2 constraint is automatically enforced at the vocabulary level: `(6/200)` does not exist as a token, only `(3/100)` does. The GCD canonicality loss is no longer needed as a training penalty — the model simply cannot emit non-canonical fractions.

**Action**: Remove A2 loss from training loop. Update the constraint zoo doc to note this. The preprocessing canonicalization (Step 1.1) does the same work at data-prep time.

### Step 3.5 — Duplicate Element Constraint (A1) Unchanged

A1 operates at the element token level, which is unchanged. Keep it as-is.

### Step 3.6 — Site Occupancy Loss (A3) Simplification

With fraction tokens, computing site occupancy sum is now cleaner — directly sum the float values of fraction tokens on the same crystal site, without needing to reconstruct the fraction from character predictions.

```python
def site_occupancy_loss(decoded_tokens, tokenizer, family):
    """
    Compute L1 loss on site occupancy sum deviation.
    Cleaner with atomic fraction tokens — no reconstruction needed.
    """
    # Parse element-fraction pairs from token sequence
    sites = parse_sites_by_family(decoded_tokens, tokenizer, family)
    
    total_loss = 0.0
    for site_tokens in sites:
        site_sum = 0.0
        for tok_id in site_tokens:
            val = tokenizer.fraction_token_to_value(tok_id)
            if val is not None:
                site_sum += val
            elif tokenizer.id_to_token[tok_id].isdigit():
                site_sum += int(tokenizer.id_to_token[tok_id])
        target_sum = get_expected_site_sum(family)  # 1.0, 2.0 etc
        total_loss += abs(site_sum - target_sum)
    
    return total_loss
```

---

## Phase 4: Training Configuration

### Step 4.1 — Loss Function Updates

**Remove** (handled by tokenization):
- `numden_loss` (regression on num/den values)
- `gcd_canonicality_loss` (A2 — impossible to emit non-canonical tokens now)

**Keep** (unchanged):
- `reconstruction_loss` (token cross-entropy, now over larger vocab)
- `tc_loss` (Tc regression from Z)
- `family_loss` (family classification)
- `sc_loss` (SC classification)
- `kl_loss` (VAE KL divergence)
- `duplicate_element_loss` (A1)
- `round_trip_loss` (A5, run on 10% of batch)

**Update**:
- `site_occupancy_loss` (A3 — use fraction_token_to_value instead of numden reconstruction)

**New** (if implementing A8 from constraint zoo):
- `structural_completeness_loss`

**New loss weight** for fraction tokens:
The cross-entropy loss naturally includes fraction positions now. Consider upweighting fraction token positions since they carry more semantic content than integer tokens:

```python
def weighted_reconstruction_loss(logits, targets, tokenizer, fraction_weight=2.0):
    """
    Cross-entropy reconstruction loss with upweighting for fraction positions.
    
    Rationale: A wrong fraction prediction is semantically worse than
    a wrong integer prediction — (17/20) vs (18/20) changes the doping
    level, which shifts Tc. Upweighting makes the model invest more
    in getting fractions right.
    """
    # Standard cross-entropy
    ce = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), 
                         targets.view(-1), 
                         reduction='none')
    
    # Build position weights
    weights = torch.ones_like(targets, dtype=torch.float)
    for i, tok_id in enumerate(targets.view(-1)):
        if tokenizer.is_fraction_token(tok_id.item()):
            weights[i] = fraction_weight
    
    return (ce * weights).mean()
```

### Step 4.2 — Checkpoint Transfer Strategy

This is the most operationally important step. Getting this right avoids wasting a full training run.

```python
def build_v13_from_v12_checkpoint(
    v12_checkpoint_path: str,
    v13_model,
    new_tokenizer: FractionAwareTokenizer,
    old_tokenizer,  # V12.41 tokenizer
):
    """
    Transfer all transferable weights from V12.41 → V13.0.
    
    What transfers (full weight copy):
        - Encoder: all transformer weights (no change)
        - Latent space: mu/logvar projections (no change)  
        - Z→decoder bridge: all weights (no change)
        - Tc head: full weights (no change)
        - Family head: full weights (no change)
        - SC head: full weights (no change, 2220-dim input unchanged)
        - hp head: full weights (no change)
        - Magpie predictor: full weights (no change)
        - Cross-head aggregation: full weights (no change)
    
    What requires partial transfer (vocab-dependent):
        - Token embedding table: copy rows for shared tokens
        - Output projection: copy rows for shared tokens
    
    What starts from scratch (new tokens):
        - Fraction token embedding rows (N_fractions new rows)
        - Fraction token output projection rows (N_fractions new rows)
    
    What is removed:
        - numden_head weights (no longer needed)
        - GCD canonicality loss weights (removed)
    """
    v12_state = torch.load(v12_checkpoint_path, map_location='cpu')
    v12_weights = v12_state['model_state_dict']
    
    v13_state = v13_model.state_dict()
    
    # Transfer all non-vocabulary-dependent weights by name
    VOCAB_DEPENDENT_KEYS = {
        'embedding.weight',          # adjust to actual key name
        'output_projection.weight',  # adjust to actual key name
        'output_projection.bias',    # adjust to actual key name
        'numden_head.weight',        # remove
        'numden_head.bias',          # remove
    }
    
    transferred_keys = []
    skipped_keys = []
    
    for key in v13_state:
        if key in VOCAB_DEPENDENT_KEYS:
            skipped_keys.append(key)
            continue
        if key in v12_weights and v12_weights[key].shape == v13_state[key].shape:
            v13_state[key] = v12_weights[key]
            transferred_keys.append(key)
        else:
            skipped_keys.append(key)
    
    # Handle embedding transfer separately (vocab-dependent)
    v13_model.load_state_dict(v13_state)
    
    # Transfer embedding rows
    v13_model.embedding.transfer_weights_from_v12(
        old_embedding=v12_weights['embedding.weight'],  # raw tensor
        old_tokenizer=old_tokenizer
    )
    
    # Transfer output projection rows  
    transfer_output_projection(
        v13_model.output_projection,
        v12_weights['output_projection.weight'],  # raw tensor
        new_tokenizer, old_tokenizer
    )
    
    print(f"Transferred: {len(transferred_keys)} parameter groups")
    print(f"New/modified: {len(skipped_keys)} parameter groups")
    print("Most of the model is warm-started. Only fraction embeddings train from scratch.")
    
    return v13_model


def get_warm_start_training_schedule():
    """
    Two-phase training schedule to avoid catastrophic forgetting.
    
    Phase A (first N steps): Freeze all transferred weights.
        Only train fraction token embeddings + output projection fraction rows.
        Goal: Let fraction representations stabilize before full training.
        Risk if skipped: The large gradient from randomly initialized fraction 
        tokens may disturb the carefully trained encoder/decoder weights.
    
    Phase B (remaining steps): Unfreeze everything.
        Standard training with all weights unfrozen.
        The fraction embeddings are now stable enough to train jointly.
    """
    return {
        'phase_a_steps': 2000,        # ~1-2 epochs depending on dataset size
        'phase_a_lr': 1e-4,           # Higher LR for new params only
        'phase_b_lr': 2e-5,           # Lower LR for fine-tuning everything
        'phase_a_frozen': [
            'encoder.*',
            'latent.*', 
            'tc_head.*',
            'family_head.*',
            'sc_head.*',
            # Freeze all but fraction embedding rows and output fraction rows
        ]
    }
```

### Step 4.3 — Fraction Token Initialization Strategy

Randomly initialized fraction embeddings will be far from the encoder's learned representation space, causing instability. A smarter initialization:

```python
def initialize_fraction_embeddings(embedding_table, tokenizer, strategy='physics'):
    """
    Initialize fraction token embeddings using physics-informed starting points.
    
    Strategy 'physics': Initialize fraction embedding as interpolation between
    the embeddings of its numerator integer and denominator integer.
    e.g., FRAC:3/20 starts as 0.15 * embed('1') + 0.85 * embed('0')
    (weighted blend toward the fractional value's nearest integer neighbors)
    
    Strategy 'random_small': Small random noise (safe baseline)
    """
    with torch.no_grad():
        lo, hi = tokenizer.fraction_range
        for frac_idx in range(lo, hi):
            tok = tokenizer.id_to_token[frac_idx]
            if not tok.startswith('FRAC:'):
                continue
            
            num, den = map(int, tok[5:].split('/'))
            frac_value = num / den
            
            if strategy == 'physics':
                # Interpolate between nearest available integer embeddings
                lower_int = max(1, int(frac_value))
                upper_int = min(tokenizer.max_integer, lower_int + 1)
                
                lower_id = tokenizer.token_to_id.get(str(lower_int))
                upper_id = tokenizer.token_to_id.get(str(upper_int))
                
                if lower_id and upper_id:
                    alpha = frac_value - lower_int
                    embedding_table.weight[frac_idx] = (
                        (1 - alpha) * embedding_table.weight[lower_id] +
                        alpha * embedding_table.weight[upper_id]
                    )
                    # Add small noise to break symmetry
                    embedding_table.weight[frac_idx] += torch.randn_like(
                        embedding_table.weight[frac_idx]
                    ) * 0.01
```

---

## Phase 5: Evaluation Protocol

### Step 5.1 — New Metrics for Fraction Generation

With atomic fraction tokens, the evaluation metrics need updating. The old numden MAE metric disappears; replace with:

```python
def compute_fraction_metrics(generated_formulas, target_formulas, tokenizer):
    """
    Evaluate fraction prediction quality with new tokenization.
    """
    results = {
        'fraction_token_exact_match': 0,  # Fraction token is exactly right
        'fraction_value_error_mean': 0,    # Mean |predicted_value - true_value|
        'formula_exact_match': 0,          # Full formula exactly right
        'formula_similarity': 0,           # Edit distance based similarity
        'fraction_oov_rate': 0,            # Rate of FRAC_UNK tokens in generation
    }
    
    n_frac_positions = 0
    total_frac_error = 0
    total_exact_frac = 0
    total_oov = 0
    
    for gen, target in zip(generated_formulas, target_formulas):
        gen_tokens = tokenizer.encode(gen, add_special_tokens=False)
        tgt_tokens = tokenizer.encode(target, add_special_tokens=False)
        
        # Align sequences (handle length differences gracefully)
        for gen_t, tgt_t in zip(gen_tokens, tgt_tokens):
            if tokenizer.is_fraction_token(tgt_t):
                n_frac_positions += 1
                
                # Exact match
                if gen_t == tgt_t:
                    total_exact_frac += 1
                
                # OOV
                if gen_t == tokenizer.FRAC_UNK_ID:
                    total_oov += 1
                    continue
                
                # Value error (even if wrong token, how far is the value?)
                gen_val = tokenizer.fraction_token_to_value(gen_t) or 0
                tgt_val = tokenizer.fraction_token_to_value(tgt_t) or 0
                total_frac_error += abs(gen_val - tgt_val)
        
        # Full formula metrics
        results['formula_exact_match'] += int(gen == target)
        results['formula_similarity'] += compute_similarity(gen, target)
    
    n = len(generated_formulas)
    if n_frac_positions > 0:
        results['fraction_token_exact_match'] = total_exact_frac / n_frac_positions
        results['fraction_value_error_mean'] = total_frac_error / n_frac_positions
        results['fraction_oov_rate'] = total_oov / n_frac_positions
    
    results['formula_exact_match'] /= n
    results['formula_similarity'] /= n
    
    return results
```

### Step 5.2 — Regression Test Against V12.41 Baseline

Before declaring V13.0 successful, verify on the 45-compound holdout set:

| Metric | V12.41 baseline | V13.0 target |
|--------|-----------------|--------------|
| Exact formula match | 26.7% | ≥ 35% |
| Formula similarity ≥ 0.99 | 86.7% | ≥ 90% |
| SC classification | 100% | ≥ 100% |
| Family classification | 91% | ≥ 88% (allow slight drop during transition) |
| Tc MAE | 0.51 K | ≤ 0.60 K (allow slight regression) |
| Fraction token exact match | N/A (new metric) | ≥ 85% |
| Fraction value error (mean) | ~0.03 (from numden MAE) | ≤ 0.02 |
| Sequence length (mean) | Record V12.41 | V13.0 ≤ V12.41 - (mean_fracs_per_formula × 5) |

---

## Phase 6: Implementation Order for Claude Code

Execute in this exact order to minimize rework:

1. **Run audit** (`scripts/audit_fractions.py`) and record vocabulary statistics
2. **Canonicalize data** (`scripts/canonicalize_fractions.py`) on all splits
3. **Re-run audit** on canonicalized data to get clean vocabulary counts
4. **Build fraction vocab** (`scripts/build_fraction_vocab.py`) with final vocabulary
5. **Convert datasets** (`scripts/convert_to_fraction_tokens.py`) — verify sequence length reduction
6. **Implement and test tokenizer** (`tokenizer/fraction_tokenizer.py`) — run `test_tokenizer()`
7. **Update model architecture** — embedding, output projection, remove numden_head
8. **Implement weight transfer** — build V13.0 from V12.41 checkpoint
9. **Update training loop** — remove numden_loss and gcd_loss; add weighted reconstruction loss
10. **Phase A training** (frozen transferred weights, only fraction params update)
11. **Phase B training** (full model, standard training)
12. **Evaluate on holdout** — compare against V12.41 baseline using regression test table

---

## Appendix A: Alternative — Fraction Decomposition Tokens

If the audit reveals > 2000 unique fractions, consider this alternative instead of a fully closed fraction vocabulary.

Rather than `FRAC:17/20` as a single token, use a two-token representation:
```
[NUM:17] [DEN:20]
```
where `NUM:k` and `DEN:k` are fixed-meaning tokens for values k=1..200.

This keeps vocabulary size small (400 tokens for numerators + denominators up to 200) while still reducing sequences from 7-char fractions to 2 tokens. It also naturally handles OOV fractions since any p/q with p,q ≤ 200 is representable.

**Tradeoff**: The model must still predict two correlated tokens correctly. But the correlation is simpler (these two tokens always come together), and the GCD constraint re-appears (need to ensure predicted NUM/DEN are coprime). So this approach is inferior to closed vocabulary if vocabulary size permits.

**Only use Appendix A if**: Audit shows > 2000 unique fractions AND max numerator or denominator > 200 in a significant fraction (>5%) of cases.

---

## Appendix B: Connection to Constraint Zoo

This refactor directly implements several constraint zoo items:

| Zoo item | How this refactor addresses it |
|---|---|
| A2 GCD canonicality | Eliminated at architecture level — vocabulary only contains canonical fractions |
| A3 Site occupancy | `fraction_token_to_value()` makes site sum computation direct |
| A4 Stoichiometric normalization | Complementary — still needed for integer coefficient GCD |
| A5 Round-trip consistency | Unchanged — still valuable for mode collapse detection |
| A8 Essential element | Unchanged — training signal approach still needed |

The sequence length reduction also indirectly helps A1 (duplicate element): shorter sequences mean fewer positions at which the model might accidentally re-emit an element.

---

*Plan version: 2026-02-18. For V12.41 → V13.0 refactor. Prerequisite: Phase 0 audit must complete before any code changes to determine vocabulary size and confirm expected sequence length reduction.*
