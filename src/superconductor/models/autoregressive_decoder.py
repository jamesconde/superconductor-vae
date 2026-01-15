"""
Autoregressive Formula Decoder for VAE.

Generates chemical formulas token-by-token from latent vectors.
This enables true novel formula generation, not just interpolation
of training samples.

Architecture:
    z (latent) -> GRU/Transformer decoder -> token sequence -> formula string

Token vocabulary (Extended for general chemistry):
    - Element symbols: H, He, Li, ..., Og (118 elements)
    - Digits: 0-9 (for stoichiometry and fractions)
    - Fraction separator: / (for exact stoichiometry like 7/10)
    - Parentheses: ( ) for grouping fractions and compounds
    - Brackets: [ ] for coordination complexes
    - Hydration: · (middle dot) for hydrates like CuSO4·5H2O
    - Charges: + - for ionic notation
    - Underscore: _ for subscript notation
    - Special tokens: <START>, <END>, <PAD>

IMPORTANT - Fraction Format (December 2025):
    Stoichiometry is now represented as FRACTIONS, not decimals.
    The preprocessed dataset uses format: Element(numerator/denominator)

    This supports arbitrarily large fractions via digit-by-digit tokenization.
    A 10-digit number like 1234567890 -> 10 tokens ['1','2','3',...,'0']

    Maximum tested: 100,000 denominator (dataset max), but unlimited in practice.

Examples (Fraction Format - Primary):
    "La(7/10)Sr(3/10)CuO4"     -> ['La', '(', '7', '/', '1', '0', ')', 'Sr', '(', '3', '/', '1', '0', ')', 'Cu', 'O', '4']
    "YBa2Cu3O(137/20)"         -> ['Y', 'Ba', '2', 'Cu', '3', 'O', '(', '1', '3', '7', '/', '2', '0', ')']
    "La(999/1000)Sr(1/1000)CuO4" -> Large fractions for trace doping
    "Ag(1/500)Al(499/500)"     -> ['Ag', '(', '1', '/', '5', '0', '0', ')', 'Al', '(', '4', '9', '9', '/', '5', '0', '0', ')']

Examples (Legacy Decimal Format - Deprecated):
    "YBa2Cu3O7"      -> ['Y', 'Ba', '2', 'Cu', '3', 'O', '7']
    "Mg0.9Al0.1B2"   -> ['Mg', '0', '.', '9', 'Al', '0', '.', '1', 'B', '2']
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import re


# Build vocabulary
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

DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Extended special tokens for general chemistry
SPECIAL = [
    '<PAD>', '<START>', '<END>',  # Control tokens
    '.',                           # Decimal point
    '(', ')',                      # Parentheses for grouping
    '[', ']',                      # Brackets for coordination complexes
    '{', '}',                      # Braces (rare, but complete set)
    '·', '•',                      # Hydration dots (middle dot and bullet)
    '+', '-',                      # Charge notation
    '_',                           # Subscript marker
    '^',                           # Superscript marker
    '/',                           # Ratio notation
    "'", '"',                      # Prime notation (for isomers)
    '*',                           # Wildcard/radical marker
]

# Build token vocabulary
VOCAB = SPECIAL + ELEMENTS[1:] + DIGITS  # Skip empty element at index 0

# Token indices for special tokens (for convenience)
PAD_TOKEN = '<PAD>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
TOKEN_TO_IDX = {token: idx for idx, token in enumerate(VOCAB)}
IDX_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_IDX.items()}

PAD_IDX = TOKEN_TO_IDX['<PAD>']
START_IDX = TOKEN_TO_IDX['<START>']
END_IDX = TOKEN_TO_IDX['<END>']
VOCAB_SIZE = len(VOCAB)


# Set of recognized special characters for fast lookup
SPECIAL_CHARS = set([
    '.', '(', ')', '[', ']', '{', '}',
    '·', '•', '+', '-', '_', '^', '/', "'", '"', '*'
])


def tokenize_formula(formula: str) -> List[str]:
    """
    Tokenize a chemical formula into elements, numbers, and special symbols.

    Handles:
        - Element symbols (H, He, Li, ... Og)
        - Digits (0-9) individually for stoichiometry
        - Fraction notation: (numerator/denominator) for exact stoichiometry
        - Parentheses, brackets, braces for grouping
        - Hydration dots (· and •)
        - Charge notation (+ and -)
        - Decimal points (legacy format)
        - Other special notation

    FRACTION FORMAT (Primary - December 2025):
        Preprocessed data uses fractions like "La(7/10)Sr(3/10)CuO4".
        Each digit is tokenized separately, enabling unlimited precision:
        "(999/1000)" -> ['(', '9', '9', '9', '/', '1', '0', '0', '0', ')']

    Examples (Fraction Format - Primary):
        "La(7/10)Sr(3/10)CuO4"  -> ['La', '(', '7', '/', '1', '0', ')', 'Sr', ...]
        "YBa2Cu3O(137/20)"      -> ['Y', 'Ba', '2', 'Cu', '3', 'O', '(', '1', '3', '7', '/', '2', '0', ')']
        "Ag(1/500)Al(499/500)"  -> Large fractions work via digit-by-digit encoding

    Examples (Legacy Decimal Format):
        "YBa2Cu3O7"      -> ['Y', 'Ba', '2', 'Cu', '3', 'O', '7']
        "Mg0.9Al0.1B2"   -> ['Mg', '0', '.', '9', 'Al', '0', '.', '1', 'B', '2']
        "Ca(OH)2"        -> ['Ca', '(', 'O', 'H', ')', '2']
        "K3C60"          -> ['K', '3', 'C', '6', '0']
        "CuSO4·5H2O"     -> ['Cu', 'S', 'O', '4', '·', '5', 'H', '2', 'O']
        "[Cu(NH3)4]2+"   -> ['[', 'Cu', '(', 'N', 'H', '3', ')', '4', ']', '2', '+']

    Args:
        formula: Chemical formula string (fraction or decimal format)

    Returns:
        List of tokens
    """
    tokens = []
    i = 0

    while i < len(formula):
        char = formula[i]

        # Try to match element (capital letter + optional lowercase)
        if char.isupper():
            # Check for two-letter element symbol
            if i + 1 < len(formula) and formula[i + 1].islower():
                two_letter = formula[i:i+2]
                # Verify it's a real element
                if two_letter in TOKEN_TO_IDX:
                    tokens.append(two_letter)
                    i += 2
                else:
                    # Single letter element
                    tokens.append(char)
                    i += 1
            else:
                # Single letter element
                tokens.append(char)
                i += 1

        # Match digit
        elif char.isdigit():
            tokens.append(char)
            i += 1

        # Match recognized special characters
        elif char in SPECIAL_CHARS:
            tokens.append(char)
            i += 1

        # Handle whitespace (skip)
        elif char.isspace():
            i += 1

        # Handle common alternative notations
        elif char == '·' or char == '•' or char == '∙':
            # Various dot characters -> middle dot
            tokens.append('·')
            i += 1

        # Skip unrecognized characters (but could log warning)
        else:
            i += 1

    return tokens


def tokenize_smiles(smiles: str) -> List[str]:
    """
    Tokenize a SMILES string for organic molecules.

    This is a basic SMILES tokenizer. For complex organic chemistry,
    consider using a dedicated library like RDKit.

    SMILES tokens include:
        - Atoms: C, N, O, S, P, F, Cl, Br, I, etc.
        - Bonds: -, =, #, :
        - Branches: (, )
        - Rings: digits 1-9, %10-%99
        - Stereochemistry: /, \\, @, @@

    Example:
        "c1ccccc1"       -> ['c', '1', 'c', 'c', 'c', 'c', 'c', '1']  (benzene)
        "CC(=O)O"        -> ['C', 'C', '(', '=', 'O', ')', 'O']  (acetic acid)
        "C60"            -> ['C', '6', '0']  (fullerene - simplified)

    Args:
        smiles: SMILES string

    Returns:
        List of tokens
    """
    tokens = []
    i = 0

    # SMILES-specific tokens
    smiles_special = set(['=', '#', ':', '/', '\\', '@', '%'])

    while i < len(smiles):
        char = smiles[i]

        # Bracketed atoms [...]
        if char == '[':
            # Find closing bracket
            j = i + 1
            while j < len(smiles) and smiles[j] != ']':
                j += 1
            if j < len(smiles):
                tokens.append(smiles[i:j+1])  # Include brackets
                i = j + 1
            else:
                tokens.append(char)
                i += 1

        # Two-letter elements (Cl, Br, etc.)
        elif char.isupper():
            if i + 1 < len(smiles) and smiles[i + 1].islower():
                two_letter = smiles[i:i+2]
                if two_letter in ['Cl', 'Br', 'Si', 'Se', 'As']:
                    tokens.append(two_letter)
                    i += 2
                else:
                    tokens.append(char)
                    i += 1
            else:
                tokens.append(char)
                i += 1

        # Lowercase aromatic atoms
        elif char.islower():
            tokens.append(char)
            i += 1

        # Digits (ring closures)
        elif char.isdigit():
            tokens.append(char)
            i += 1

        # Ring closure %nn
        elif char == '%':
            if i + 2 < len(smiles) and smiles[i+1:i+3].isdigit():
                tokens.append(smiles[i:i+3])
                i += 3
            else:
                tokens.append(char)
                i += 1

        # Special SMILES characters
        elif char in smiles_special or char in '()+-':
            tokens.append(char)
            i += 1

        else:
            i += 1

    return tokens


def tokens_to_indices(tokens: List[str], max_len: int = 50) -> torch.Tensor:
    """Convert token list to index tensor with padding."""
    indices = [START_IDX]
    for token in tokens:
        if token in TOKEN_TO_IDX:
            indices.append(TOKEN_TO_IDX[token])
    indices.append(END_IDX)

    # Pad or truncate
    if len(indices) < max_len:
        indices = indices + [PAD_IDX] * (max_len - len(indices))
    else:
        indices = indices[:max_len-1] + [END_IDX]

    return torch.tensor(indices, dtype=torch.long)


def indices_to_formula(indices: torch.Tensor) -> str:
    """Convert index tensor back to formula string."""
    tokens = []
    for idx in indices:
        idx = idx.item() if isinstance(idx, torch.Tensor) else idx
        if idx == END_IDX:
            break
        if idx not in [PAD_IDX, START_IDX]:
            tokens.append(IDX_TO_TOKEN.get(idx, ''))
    return ''.join(tokens)


class AutoregressiveFormulaDecoder(nn.Module):
    """
    GRU-based autoregressive decoder for generating chemical formulas.

    Takes a latent vector z and generates a formula token by token.
    Uses teacher forcing during training and greedy/sampling during inference.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 50
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = VOCAB_SIZE

        # Token embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=hidden_dim,
            padding_idx=PAD_IDX
        )

        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        # GRU decoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, VOCAB_SIZE)
        )

    def init_hidden(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize GRU hidden state from latent vector."""
        batch_size = z.size(0)
        # Project z to hidden dimensions
        h = self.latent_to_hidden(z)  # (batch, hidden * num_layers)
        # Reshape to (num_layers, batch, hidden)
        h = h.view(batch_size, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        return h

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional teacher forcing.

        Args:
            z: Latent vectors (batch, latent_dim)
            target_tokens: Target token indices for teacher forcing (batch, seq_len)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            logits: Token logits (batch, seq_len, vocab_size)
            generated: Generated token indices (batch, seq_len)
        """
        batch_size = z.size(0)
        device = z.device

        # Initialize hidden state from latent
        hidden = self.init_hidden(z)

        # Determine sequence length
        if target_tokens is not None:
            max_len = target_tokens.size(1)
        else:
            max_len = self.max_len

        # Storage for outputs
        all_logits = []
        all_tokens = []

        # Start with START token
        current_token = torch.full(
            (batch_size, 1), START_IDX, dtype=torch.long, device=device
        )

        for t in range(max_len - 1):
            # Embed current token
            embedded = self.token_embedding(current_token)  # (batch, 1, hidden)

            # GRU step
            output, hidden = self.gru(embedded, hidden)

            # Project to vocabulary
            logits = self.output_proj(output)  # (batch, 1, vocab_size)
            all_logits.append(logits)

            # Get next token (greedy or from target)
            if target_tokens is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                current_token = target_tokens[:, t+1:t+2]
            else:
                # Greedy decoding
                current_token = logits.argmax(dim=-1)

            all_tokens.append(current_token)

        # Concatenate outputs
        logits = torch.cat(all_logits, dim=1)  # (batch, seq_len-1, vocab_size)
        generated = torch.cat(all_tokens, dim=1)  # (batch, seq_len-1)

        return logits, generated

    def generate(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_len: Optional[int] = None
    ) -> List[str]:
        """
        Generate formulas from latent vectors.

        Args:
            z: Latent vectors (batch, latent_dim)
            temperature: Sampling temperature (1.0 = normal, <1 = more deterministic)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling with this probability mass
            max_len: Maximum sequence length

        Returns:
            List of generated formula strings
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        with torch.no_grad():
            hidden = self.init_hidden(z)
            current_token = torch.full(
                (batch_size, 1), START_IDX, dtype=torch.long, device=device
            )

            generated_tokens = []
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                embedded = self.token_embedding(current_token)
                output, hidden = self.gru(embedded, hidden)
                logits = self.output_proj(output).squeeze(1)  # (batch, vocab_size)

                # Apply temperature
                logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for END token
                finished = finished | (next_token.squeeze(-1) == END_IDX)

                generated_tokens.append(next_token)
                current_token = next_token

                if finished.all():
                    break

            # Convert to formulas
            all_tokens = torch.cat(generated_tokens, dim=1)
            formulas = [indices_to_formula(all_tokens[i]) for i in range(batch_size)]

        return formulas

    def compute_loss(
        self,
        z: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            z: Latent vectors (batch, latent_dim)
            target_tokens: Target token indices (batch, seq_len)
            teacher_forcing_ratio: Probability of teacher forcing

        Returns:
            Dictionary with loss values
        """
        logits, _ = self.forward(z, target_tokens, teacher_forcing_ratio)

        # Target is shifted by 1 (predict next token)
        target = target_tokens[:, 1:logits.size(1)+1]

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, self.vocab_size)
        target_flat = target.reshape(-1)

        # Ignore padding tokens in loss
        loss = F.cross_entropy(
            logits_flat, target_flat,
            ignore_index=PAD_IDX,
            reduction='mean'
        )

        # Compute accuracy (ignoring padding)
        predictions = logits.argmax(dim=-1)
        mask = target != PAD_IDX
        correct = ((predictions == target) & mask).sum()
        total = mask.sum()
        accuracy = correct.float() / (total.float() + 1e-8)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'perplexity': torch.exp(loss)
        }


class FormulaVAEWithDecoder(nn.Module):
    """
    VAE with autoregressive formula decoder.

    Combines the attention-based encoder with the autoregressive decoder
    for end-to-end training and generation.
    """

    def __init__(
        self,
        encoder,  # AttentionBidirectionalVAE or similar
        decoder: AutoregressiveFormulaDecoder,
        reconstruction_weight: float = 1.0
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        isotope_features: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ):
        """
        Forward pass through encoder and decoder.
        """
        # Encode to latent
        encoder_outputs = self.encoder(
            element_indices=element_indices,
            element_fractions=element_fractions,
            element_mask=element_mask,
            isotope_features=isotope_features,
            return_all=True
        )

        z = encoder_outputs['z']

        # Decode to formula
        if target_tokens is not None:
            decoder_loss = self.decoder.compute_loss(z, target_tokens, teacher_forcing_ratio)
        else:
            decoder_loss = None

        return {
            **encoder_outputs,
            'decoder_loss': decoder_loss
        }

    def generate_formulas(
        self,
        z: torch.Tensor = None,
        n_samples: int = 10,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate novel formulas.

        Args:
            z: Latent vectors to decode (if None, sample from prior)
            n_samples: Number of samples if z is None
            temperature: Sampling temperature

        Returns:
            List of generated formula strings
        """
        device = next(self.parameters()).device

        if z is None:
            # Sample from prior (standard normal)
            z = torch.randn(n_samples, self.encoder.latent_dim, device=device)

        return self.decoder.generate(z, temperature=temperature, **kwargs)


def create_formula_tokenizer():
    """Create tokenization utilities for formula processing."""
    return {
        'tokenize': tokenize_formula,
        'to_indices': tokens_to_indices,
        'to_formula': indices_to_formula,
        'vocab_size': VOCAB_SIZE,
        'pad_idx': PAD_IDX,
        'start_idx': START_IDX,
        'end_idx': END_IDX,
    }


# =============================================================================
# Transformer-based Decoder (Higher capacity for 100% reconstruction)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFormulaDecoder(nn.Module):
    """
    Transformer-based autoregressive decoder for chemical formulas.

    Uses self-attention over generated tokens and cross-attention to the
    latent vector z for conditioning. Much higher capacity than GRU for
    achieving near-perfect reconstruction.

    Architecture:
        - Token embedding + positional encoding
        - Latent projection to memory sequence
        - N transformer decoder layers with:
            - Masked self-attention (causal)
            - Cross-attention to latent memory
            - Feed-forward network
        - Output projection to vocabulary

    Args:
        latent_dim: Dimension of input latent vectors
        d_model: Transformer hidden dimension (default: 512)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        dim_feedforward: FFN hidden dimension (default: 2048)
        dropout: Dropout rate (default: 0.1)
        max_len: Maximum sequence length (default: 50)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 50
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = VOCAB_SIZE

        # Token embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=d_model,
            padding_idx=PAD_IDX
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Project latent to memory for cross-attention
        # We create a small "memory" sequence from z for the decoder to attend to
        self.latent_to_memory = nn.Sequential(
            nn.Linear(latent_dim, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 8),  # 8 memory tokens
        )
        self.n_memory_tokens = 8

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, VOCAB_SIZE)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _create_memory(self, z: torch.Tensor) -> torch.Tensor:
        """Create memory sequence from latent vector for cross-attention."""
        batch_size = z.size(0)
        # Project z to memory tokens
        memory = self.latent_to_memory(z)  # (batch, d_model * n_memory)
        memory = memory.view(batch_size, self.n_memory_tokens, self.d_model)
        return memory

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 1.0  # Not used, kept for API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing.

        Args:
            z: Latent vectors (batch, latent_dim)
            target_tokens: Target token indices (batch, seq_len)
            teacher_forcing_ratio: Ignored (Transformer always uses full teacher forcing)

        Returns:
            logits: Token logits (batch, seq_len-1, vocab_size)
            generated: Predicted token indices (batch, seq_len-1)
        """
        batch_size = z.size(0)
        device = z.device

        # Create memory from latent
        memory = self._create_memory(z)  # (batch, n_memory, d_model)

        # Embed input tokens (all but last, since we predict next token)
        input_tokens = target_tokens[:, :-1]  # (batch, seq_len-1)
        embedded = self.token_embedding(input_tokens)  # (batch, seq_len-1, d_model)
        embedded = self.pos_encoding(embedded)

        # Create causal mask
        seq_len = input_tokens.size(1)
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Create padding mask for input
        tgt_key_padding_mask = (input_tokens == PAD_IDX)

        # Transformer decoder
        output = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Project to vocabulary
        logits = self.output_proj(output)  # (batch, seq_len-1, vocab_size)

        # Get predictions
        generated = logits.argmax(dim=-1)

        return logits, generated

    def generate(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_len: Optional[int] = None
    ) -> List[str]:
        """
        Generate formulas autoregressively from latent vectors.

        Args:
            z: Latent vectors (batch, latent_dim)
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling
            max_len: Maximum sequence length

        Returns:
            List of generated formula strings
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        with torch.no_grad():
            # Create memory from latent
            memory = self._create_memory(z)

            # Start with START token
            generated_tokens = torch.full(
                (batch_size, 1), START_IDX, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(max_len - 1):
                # Embed current sequence
                embedded = self.token_embedding(generated_tokens)
                embedded = self.pos_encoding(embedded)

                # Create causal mask
                seq_len = generated_tokens.size(1)
                causal_mask = self._generate_causal_mask(seq_len, device)

                # Transformer forward
                output = self.transformer_decoder(
                    tgt=embedded,
                    memory=memory,
                    tgt_mask=causal_mask
                )

                # Get logits for last position only
                logits = self.output_proj(output[:, -1, :])  # (batch, vocab_size)

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample or argmax
                if temperature < 0.01:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                # Update finished status
                finished = finished | (next_token.squeeze(-1) == END_IDX)

                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                if finished.all():
                    break

            # Convert to formulas
            formulas = []
            for i in range(batch_size):
                formula = indices_to_formula(generated_tokens[i, 1:])  # Skip START
                formulas.append(formula)

        return formulas

    def generate_indices(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate token indices autoregressively from latent vectors.

        Similar to generate() but returns raw token indices instead of formula strings.

        Args:
            z: Latent vectors (batch, latent_dim)
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling
            max_len: Maximum sequence length

        Returns:
            Token indices tensor (batch, seq_len) - excludes START token
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        with torch.no_grad():
            # Create memory from latent
            memory = self._create_memory(z)

            # Start with START token
            generated_tokens = torch.full(
                (batch_size, 1), START_IDX, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(max_len - 1):
                # Embed current sequence
                embedded = self.token_embedding(generated_tokens)
                embedded = self.pos_encoding(embedded)

                # Create causal mask
                seq_len = generated_tokens.size(1)
                causal_mask = self._generate_causal_mask(seq_len, device)

                # Transformer forward
                output = self.transformer_decoder(
                    tgt=embedded,
                    memory=memory,
                    tgt_mask=causal_mask
                )

                # Get logits for last position only
                logits = self.output_proj(output[:, -1, :])  # (batch, vocab_size)

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample or argmax
                if temperature < 0.01:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                # Update finished status
                finished = finished | (next_token.squeeze(-1) == END_IDX)

                # Append to generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                if finished.all():
                    break

        # Return token indices excluding START token
        return generated_tokens[:, 1:]

    def compute_loss(
        self,
        z: torch.Tensor,
        target_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 1.0  # Ignored
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            z: Latent vectors (batch, latent_dim)
            target_tokens: Target token indices (batch, seq_len)
            teacher_forcing_ratio: Ignored for Transformer

        Returns:
            Dictionary with loss, accuracy, perplexity
        """
        logits, _ = self.forward(z, target_tokens)

        # Target is the tokens shifted by 1 (predict next token)
        target = target_tokens[:, 1:logits.size(1)+1]

        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, self.vocab_size)
        target_flat = target.reshape(-1)

        # Compute loss ignoring padding
        loss = F.cross_entropy(
            logits_flat, target_flat,
            ignore_index=PAD_IDX,
            reduction='mean',
            label_smoothing=0.1  # Add label smoothing for better generalization
        )

        # Compute accuracy (ignoring padding)
        predictions = logits.argmax(dim=-1)
        mask = target != PAD_IDX
        correct = ((predictions == target) & mask).sum()
        total = mask.sum()
        accuracy = correct.float() / (total.float() + 1e-8)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'perplexity': torch.exp(loss)
        }


class FormulaVAEWithTransformer(nn.Module):
    """
    VAE with Transformer formula decoder.

    Higher capacity version for near-perfect reconstruction.
    """

    def __init__(
        self,
        encoder,
        decoder: TransformerFormulaDecoder,
        reconstruction_weight: float = 1.0
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        element_indices: torch.Tensor,
        element_fractions: torch.Tensor,
        element_mask: torch.Tensor,
        isotope_features: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None
    ):
        """Forward pass through encoder and decoder."""
        # Encode
        encoder_outputs = self.encoder(
            element_indices=element_indices,
            element_fractions=element_fractions,
            element_mask=element_mask,
            isotope_features=isotope_features,
            return_all=True
        )

        z = encoder_outputs['z']

        # Decode
        if target_tokens is not None:
            decoder_loss = self.decoder.compute_loss(z, target_tokens)
        else:
            decoder_loss = None

        return {
            **encoder_outputs,
            'decoder_loss': decoder_loss
        }

    def generate_formulas(
        self,
        z: torch.Tensor = None,
        n_samples: int = 10,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """Generate novel formulas."""
        device = next(self.parameters()).device

        if z is None:
            z = torch.randn(n_samples, self.encoder.latent_dim, device=device)

        return self.decoder.generate(z, temperature=temperature, **kwargs)


# =============================================================================
# Fraction Tokenization Verification
# =============================================================================

def verify_fraction_tokenization(formula: str, verbose: bool = False) -> bool:
    """
    Verify that a fraction-format formula tokenizes and detokenizes correctly.

    This ensures exact round-trip conversion for fraction notation:
        "La(7/10)Sr(3/10)CuO4" -> tokens -> indices -> formula -> "La(7/10)Sr(3/10)CuO4"

    Args:
        formula: Chemical formula with fraction notation
        verbose: Print debug information

    Returns:
        True if round-trip is exact, False otherwise
    """
    # Tokenize
    tokens = tokenize_formula(formula)

    # Convert to indices
    indices = tokens_to_indices(tokens)

    # Convert back to formula
    reconstructed = indices_to_formula(indices)

    # Check exact match
    success = reconstructed == formula

    if verbose:
        print(f"Original:      {formula}")
        print(f"Tokens:        {tokens}")
        print(f"Reconstructed: {reconstructed}")
        print(f"Match: {success}")

    return success


def verify_large_fraction_support(max_digits: int = 10) -> bool:
    """
    Verify that the tokenizer can handle large fractions.

    Tests that fractions with up to max_digits digits in numerator/denominator
    tokenize correctly.

    Args:
        max_digits: Maximum number of digits to test

    Returns:
        True if all tests pass
    """
    test_cases = [
        "La(7/10)Sr(3/10)CuO4",           # Simple fractions
        "La(999/1000)Sr(1/1000)CuO4",     # 3-digit fractions
        "Ag(1/10000)Al(9999/10000)",      # 4-digit fractions
        "X(12345/67890)",                  # 5-digit fractions
        "Y(123456/789012)",                # 6-digit fractions
    ]

    # Add test case with max_digits
    if max_digits >= 10:
        big_num = "1234567890"
        test_cases.append(f"Z({big_num}/{big_num})")

    all_passed = True
    for formula in test_cases:
        tokens = tokenize_formula(formula)
        indices = tokens_to_indices(tokens, max_len=100)  # Longer max_len for big fractions
        reconstructed = indices_to_formula(indices)

        if reconstructed != formula:
            print(f"FAIL: {formula} -> {reconstructed}")
            all_passed = False

    return all_passed


def get_tokenization_stats(formulas: List[str]) -> Dict:
    """
    Get tokenization statistics for a list of formulas.

    Useful for analyzing the preprocessed fraction dataset.

    Args:
        formulas: List of formula strings

    Returns:
        Dictionary with tokenization statistics
    """
    stats = {
        'total_formulas': len(formulas),
        'total_tokens': 0,
        'max_tokens': 0,
        'min_tokens': float('inf'),
        'avg_tokens': 0,
        'token_distribution': {},
        'round_trip_success': 0,
    }

    for formula in formulas:
        tokens = tokenize_formula(formula)
        n_tokens = len(tokens)

        stats['total_tokens'] += n_tokens
        stats['max_tokens'] = max(stats['max_tokens'], n_tokens)
        stats['min_tokens'] = min(stats['min_tokens'], n_tokens)

        # Track token distribution
        for token in tokens:
            if token not in stats['token_distribution']:
                stats['token_distribution'][token] = 0
            stats['token_distribution'][token] += 1

        # Test round-trip
        if verify_fraction_tokenization(formula, verbose=False):
            stats['round_trip_success'] += 1

    stats['avg_tokens'] = stats['total_tokens'] / len(formulas) if formulas else 0
    stats['round_trip_rate'] = stats['round_trip_success'] / len(formulas) if formulas else 0

    return stats


# ============================================================================
# V11 ENHANCED DECODER - MORE CAPACITY + SKIP CONNECTIONS
# ============================================================================

class EnhancedTransformerDecoder(nn.Module):
    """
    V11 Enhanced Transformer Decoder with skip connections.

    Key improvements over TransformerFormulaDecoder:
    1. More memory tokens (16 vs 8) for richer latent representation
    2. Skip connections from encoder (element representations)
    3. Configurable architecture for scaling

    The skip connections break "pure VAE" but enable direct information flow
    from encoder element representations to decoder, which is what we want
    for learning a shared theory of superconductivity.

    Architecture:
        Latent z (2048) → 16 memory tokens (512-dim each)
        Encoder skip (element_repr) → projected to d_model
        Combined memory = [latent_memory; skip_memory]
        Decoder cross-attends to combined memory
    """

    def __init__(
        self,
        latent_dim: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 80,
        n_memory_tokens: int = 16,
        encoder_skip_dim: int = 256,  # Dimension of encoder's element representation
        use_skip_connection: bool = True,
        # V12.4: Stoichiometry conditioning
        use_stoich_conditioning: bool = True,
        max_elements: int = 12,  # Maximum elements per formula
        n_stoich_tokens: int = 4,  # Number of stoichiometry memory tokens
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = VOCAB_SIZE
        self.n_memory_tokens = n_memory_tokens
        self.use_skip_connection = use_skip_connection
        self.use_stoich_conditioning = use_stoich_conditioning
        self.max_elements = max_elements

        # Token embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=d_model,
            padding_idx=PAD_IDX
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Project latent to memory for cross-attention (MORE tokens)
        self.latent_to_memory = nn.Sequential(
            nn.Linear(latent_dim, d_model * n_memory_tokens // 2),
            nn.GELU(),
            nn.Linear(d_model * n_memory_tokens // 2, d_model * n_memory_tokens),
        )

        # Skip connection: project encoder representation to memory tokens
        if use_skip_connection:
            # Encoder provides element_repr of shape [batch, encoder_skip_dim]
            # We create additional memory tokens from this
            self.skip_n_tokens = 8  # Additional skip tokens
            self.skip_to_memory = nn.Sequential(
                nn.Linear(encoder_skip_dim, d_model * self.skip_n_tokens // 2),
                nn.GELU(),
                nn.Linear(d_model * self.skip_n_tokens // 2, d_model * self.skip_n_tokens),
            )
        else:
            self.skip_n_tokens = 0

        # V12.4: Stoichiometry conditioning - project fraction predictions to memory tokens
        # This gives the decoder DIRECT visibility into predicted element fractions,
        # so when generating digits it can "look at" what stoichiometry it should produce.
        # Input: fraction_pred [batch, max_elements] - predicted fractions for each element
        # Output: stoich_memory [batch, n_stoich_tokens, d_model]
        if use_stoich_conditioning:
            self.stoich_n_tokens = n_stoich_tokens
            # Project max_elements fractions to memory tokens
            # Include element count prediction (+1) for richer conditioning
            stoich_input_dim = max_elements + 1  # fractions + count
            self.stoich_to_memory = nn.Sequential(
                nn.Linear(stoich_input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model * self.stoich_n_tokens),
            )
        else:
            self.stoich_n_tokens = 0

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, VOCAB_SIZE)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _create_memory(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,  # V12.4: [batch, max_elements + 1]
    ) -> torch.Tensor:
        """Create combined memory from latent z, skip connection, and stoichiometry.

        V12.4: Now includes stoichiometry conditioning tokens that give the decoder
        direct visibility into predicted element fractions. This helps with digit
        generation because the decoder can "look at" what stoichiometry to produce.

        Memory layout: [latent_tokens (16) | skip_tokens (8) | stoich_tokens (4)]
        Total: 28 memory tokens for cross-attention
        """
        batch_size = z.size(0)

        # Latent memory tokens (16 tokens)
        latent_memory = self.latent_to_memory(z)
        latent_memory = latent_memory.view(batch_size, self.n_memory_tokens, self.d_model)

        memory_parts = [latent_memory]

        # Add skip connection memory if provided (8 tokens)
        if self.use_skip_connection and encoder_skip is not None:
            skip_memory = self.skip_to_memory(encoder_skip)
            skip_memory = skip_memory.view(batch_size, self.skip_n_tokens, self.d_model)
            memory_parts.append(skip_memory)

        # V12.4: Add stoichiometry conditioning memory (4 tokens)
        # This gives decoder direct access to predicted element fractions
        if self.use_stoich_conditioning and stoich_pred is not None:
            stoich_memory = self.stoich_to_memory(stoich_pred)
            stoich_memory = stoich_memory.view(batch_size, self.stoich_n_tokens, self.d_model)
            memory_parts.append(stoich_memory)

        # Concatenate all memory parts
        memory = torch.cat(memory_parts, dim=1)

        return memory

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
        stoich_pred: Optional[torch.Tensor] = None,  # V12.4: [batch, max_elements + 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with scheduled sampling (teacher forcing).

        V12.5 FIX: Now actually implements autoregressive training when TF < 1.0.
        Previously this always used ground truth tokens, making training artificially easy.

        Args:
            z: Latent vectors (batch, latent_dim)
            target_tokens: Target token indices (batch, seq_len)
            encoder_skip: Skip connection from encoder (batch, encoder_skip_dim)
            teacher_forcing_ratio: Probability of using ground truth at each position.
                                   1.0 = always ground truth (efficient parallel forward)
                                   0.0 = always use model predictions (true autoregressive)
                                   0.5 = 50% chance of using model prediction at each step
            stoich_pred: V12.4 stoichiometry conditioning [batch, max_elements + 1]
                        Contains predicted element fractions + element count

        Returns:
            logits: Token logits (batch, seq_len-1, vocab_size)
            generated: Predicted token indices (batch, seq_len-1)
        """
        batch_size = z.size(0)
        device = z.device
        seq_len = target_tokens.size(1) - 1  # -1 because we predict next token

        # Create combined memory (V12.4: includes stoichiometry tokens)
        memory = self._create_memory(z, encoder_skip, stoich_pred)

        # Fast path: Full teacher forcing (TF = 1.0) - parallel forward
        if teacher_forcing_ratio >= 1.0:
            input_tokens = target_tokens[:, :-1]
            embedded = self.token_embedding(input_tokens)
            embedded = self.pos_encoding(embedded)

            causal_mask = self._generate_causal_mask(seq_len, device)
            tgt_key_padding_mask = (input_tokens == PAD_IDX)

            output = self.transformer_decoder(
                tgt=embedded,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            logits = self.output_proj(output)
            generated = logits.argmax(dim=-1)
            return logits, generated

        # Slow path: Scheduled sampling (TF < 1.0) - sequential autoregressive
        # This is necessary to train the model to handle its own predictions
        # NOTE: Matches generate() method structure to avoid CUDA/autocast issues
        all_logits = []
        all_generated = []

        # Start with START token
        current_tokens = target_tokens[:, :1]  # [batch, 1] = START token

        for t in range(seq_len):
            # Embed current sequence
            embedded = self.token_embedding(current_tokens)
            embedded = self.pos_encoding(embedded)

            # Create causal mask for current length
            curr_len = current_tokens.size(1)
            causal_mask = self._generate_causal_mask(curr_len, device)

            # Transformer forward (no tgt_key_padding_mask - matches generate())
            output = self.transformer_decoder(
                tgt=embedded,
                memory=memory,
                tgt_mask=causal_mask
            )

            # Get logits for last position only
            last_logits = self.output_proj(output[:, -1:, :])  # [batch, 1, vocab]
            all_logits.append(last_logits)

            # Sample next token
            pred_token = last_logits.argmax(dim=-1)  # [batch, 1]
            all_generated.append(pred_token)

            # Scheduled sampling: decide whether to use ground truth or prediction
            if t < seq_len - 1:  # Don't need next token for last position
                gt_token = target_tokens[:, t + 1:t + 2]  # [batch, 1] ground truth

                # Per-sample random choice: use GT with probability teacher_forcing_ratio
                use_gt = (torch.rand(batch_size, 1, device=device) < teacher_forcing_ratio).long()
                next_token = use_gt * gt_token + (1 - use_gt) * pred_token

                # Append to sequence
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

        # Concatenate all timesteps
        logits = torch.cat(all_logits, dim=1)  # [batch, seq_len, vocab]
        generated = torch.cat(all_generated, dim=1)  # [batch, seq_len]

        return logits, generated

    def generate(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_len: Optional[int] = None
    ) -> List[str]:
        """Generate formulas autoregressively."""
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        with torch.no_grad():
            memory = self._create_memory(z, encoder_skip)

            generated_tokens = torch.full(
                (batch_size, 1), START_IDX, dtype=torch.long, device=device
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(max_len - 1):
                embedded = self.token_embedding(generated_tokens)
                embedded = self.pos_encoding(embedded)

                seq_len = generated_tokens.size(1)
                causal_mask = self._generate_causal_mask(seq_len, device)

                output = self.transformer_decoder(
                    tgt=embedded,
                    memory=memory,
                    tgt_mask=causal_mask
                )

                logits = self.output_proj(output[:, -1, :])

                if temperature != 1.0:
                    logits = logits / temperature

                if temperature < 0.01:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                finished = finished | (next_token.squeeze(-1) == END_IDX)
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                if finished.all():
                    break

            formulas = []
            for i in range(batch_size):
                formula = indices_to_formula(generated_tokens[i, 1:])
                formulas.append(formula)

        return formulas
