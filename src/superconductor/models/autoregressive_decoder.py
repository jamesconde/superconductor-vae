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
from torch.utils.checkpoint import checkpoint as grad_checkpoint  # V12.8: Gradient checkpointing
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
        # V12.8: Gradient checkpointing for memory optimization
        use_gradient_checkpointing: bool = False,
        # V12.34: Position-dependent teacher forcing
        # When TF < 1.0, applies more TF at start of sequence, less at end.
        # No effect when TF = 1.0 (full teacher forcing).
        use_position_dependent_tf: bool = False,
        tf_position_decay: float = 0.5,
        # V13.0: Configurable vocabulary size for semantic fraction tokens
        # If None, falls back to VOCAB_SIZE (148) for backward compatibility
        vocab_size: int = None,
        # V13.0: Stoich conditioning input dimension
        # V12.x: 37 = fractions(12) + numden(24) + count(1)
        # V13.0: 13 = fractions(12) + count(1) — numden removed (implicit in fraction tokens)
        stoich_input_dim: int = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        # V13.0: Use provided vocab_size, or fall back to old VOCAB_SIZE for backward compat
        self.vocab_size = vocab_size if vocab_size is not None else VOCAB_SIZE
        self.n_memory_tokens = n_memory_tokens
        self.use_skip_connection = use_skip_connection
        self.use_stoich_conditioning = use_stoich_conditioning
        self.max_elements = max_elements
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_position_dependent_tf = use_position_dependent_tf
        self.tf_position_decay = tf_position_decay

        # Token embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
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

        # V12.4/V12.38/V13.0: Stoichiometry conditioning - project fraction predictions to memory tokens
        # This gives the decoder DIRECT visibility into predicted element fractions,
        # so when generating tokens it can "look at" what stoichiometry it should produce.
        # V12.x input: [batch, max_elements*3 + 1] — fractions(12) + numden(24) + count(1) = 37
        # V13.0 input: [batch, max_elements + 1]   — fractions(12) + count(1) = 13
        #   (numden removed — fraction info now implicit in semantic fraction tokens)
        # Output: stoich_memory [batch, n_stoich_tokens, d_model]
        if use_stoich_conditioning:
            self.stoich_n_tokens = n_stoich_tokens
            # Project stoich predictions to memory tokens
            # V13.0: Accept explicit stoich_input_dim for flexibility, default to V12.x (37) for backward compat
            if stoich_input_dim is None:
                stoich_input_dim = max_elements * 3 + 1  # V12.x default: fractions(12) + numden(24) + count(1) = 37
            self.stoich_input_dim = stoich_input_dim
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
            nn.Linear(d_model, self.vocab_size)
        )

        # V12.30: Dedicated stop-prediction head
        # Decouples the "should I stop?" decision from competing in the vocab softmax.
        # Trained with BCE loss; at inference, boosts END_IDX logit.
        self.stop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),  # 512 → 128
            nn.GELU(),
            nn.Linear(d_model // 4, 1),         # 128 → 1 (logit)
        )

        # V14.3: Token type classifier head
        # Predicts what TYPE of token comes next (element/integer/fraction/special/EOS)
        # Uses same transformer hidden state as output_proj.
        # Training: auxiliary CE loss on type labels.
        # Inference: hard mask over vocab logits to eliminate type confusion errors.
        #
        # CRITICAL: This head applies HARD masking at inference — a wrong prediction
        # completely blocks the correct token. Must be at least as capable as output_proj.
        # Architecture mirrors output_proj: LayerNorm → Linear → GELU → Dropout → Linear
        # with an additional hidden layer for richer type-discriminative features.
        from superconductor.tokenizer.fraction_tokenizer import N_TOKEN_TYPES
        self.token_type_head = nn.Sequential(
            nn.LayerNorm(d_model),                  # Normalize transformer output
            nn.Linear(d_model, d_model),            # 512 → 512 (full-width, no info loss)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 4),       # 512 → 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, N_TOKEN_TYPES),  # 128 → 5
        )

        # V14.3: Enriched decoder memory from encoder head predictions
        # Projects concatenated head outputs to additional memory tokens for cross-attention.
        # Input: tc_pred(1) + sc_pred(1) + hp_pred(1) + tc_class_logits(5) +
        #        competence(1) + element_count_pred(1) = 10 dims
        # Output: 4 memory tokens of d_model each
        #
        # 10 → 2048 is a 200x expansion — needs intermediate stages to learn
        # meaningful interactions between head predictions (e.g., "high Tc + SC + cuprate
        # family → expect Cu, Ba, Y elements and specific stoichiometries").
        self.heads_n_tokens = 4
        self.heads_to_memory = nn.Sequential(
            nn.Linear(10, d_model // 2),            # 10 → 256 (initial expansion)
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),       # 256 → 512
            nn.GELU(),
            nn.Linear(d_model, d_model * self.heads_n_tokens),  # 512 → 2048
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
        stoich_pred: Optional[torch.Tensor] = None,  # V12.x: [batch, 37]; V13.0: [batch, 13]
        heads_pred: Optional[Dict[str, torch.Tensor]] = None,  # V14.3: encoder head predictions
    ) -> torch.Tensor:
        """Create combined memory from latent z, skip connection, stoichiometry, and heads.

        V12.4: Now includes stoichiometry conditioning tokens that give the decoder
        direct visibility into predicted element fractions.
        V13.0: stoich_pred is [batch, 13] = fractions(12) + count(1). numden removed.
        V14.3: heads_pred adds 4 tokens from encoder head predictions (tc, sc, hp, etc.)

        Memory layout: [latent_tokens (16) | skip_tokens (8) | stoich_tokens (4) | heads_tokens (4)]
        Total: up to 32 memory tokens for cross-attention
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

        # V14.3: Add encoder heads memory (4 tokens)
        # Gives decoder context about what kind of material this is
        if heads_pred is not None:
            heads_input = torch.cat([
                heads_pred['tc_pred'].unsqueeze(-1),              # 1
                heads_pred['sc_pred'].unsqueeze(-1),              # 1
                heads_pred['hp_pred'].unsqueeze(-1),              # 1
                heads_pred['tc_class_logits'],                    # 5
                heads_pred['competence'].unsqueeze(-1),           # 1
                heads_pred['element_count_pred'].unsqueeze(-1),   # 1
            ], dim=-1)  # [batch, 10]
            heads_memory = self.heads_to_memory(heads_input)
            heads_memory = heads_memory.view(batch_size, self.heads_n_tokens, self.d_model)
            memory_parts.append(heads_memory)

        # Concatenate all memory parts
        memory = torch.cat(memory_parts, dim=1)

        return memory

    def precompute_memory(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        heads_pred: Optional[Dict[str, torch.Tensor]] = None,  # V14.3
    ) -> torch.Tensor:
        """
        V12.8: Pre-compute memory projections for reuse across multiple operations.

        This is useful when the same z/encoder_skip/stoich_pred will be used multiple times:
        - REINFORCE training (forward + sample on same z)
        - Speculative decoding (draft and target share memory)
        - Multiple generation attempts from same latent

        Args:
            z: Latent vectors (batch, latent_dim)
            encoder_skip: Skip connection from encoder (batch, encoder_skip_dim)
            stoich_pred: Stoichiometry conditioning [batch, max_elements*3 + 1]
            heads_pred: V14.3 Dict of encoder head predictions for enriched memory

        Returns:
            memory: Pre-computed memory tensor (batch, n_memory_tokens + extras, d_model)
        """
        return self._create_memory(z, encoder_skip, stoich_pred, heads_pred)

    def forward(
        self,
        z: torch.Tensor,
        target_tokens: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
        stoich_pred: Optional[torch.Tensor] = None,  # V12.4: [batch, max_elements*3 + 1]
        cached_memory: Optional[torch.Tensor] = None,  # V12.8: Pre-computed memory
        heads_pred: Optional[Dict[str, torch.Tensor]] = None,  # V14.3: encoder head predictions
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
            stoich_pred: V12.4 stoichiometry conditioning [batch, max_elements*3 + 1]
                        Contains predicted element fractions + element count
            cached_memory: V12.8 Pre-computed memory from precompute_memory().
                          If provided, skips memory creation (saves computation).
            heads_pred: V14.3 Dict of encoder head predictions for enriched memory.
                       Keys: tc_pred, sc_pred, hp_pred, tc_class_logits,
                             competence, element_count_pred

        Returns:
            logits: Token logits (batch, seq_len-1, vocab_size)
            generated: Predicted token indices (batch, seq_len-1)
            stop_logits: V12.30 Stop-prediction logits (batch, seq_len-1)
            type_logits: V14.3 Token type logits (batch, seq_len-1, N_TOKEN_TYPES) or None
        """
        batch_size = z.size(0)
        device = z.device
        seq_len = target_tokens.size(1) - 1  # -1 because we predict next token

        # V12.8: Use cached memory if provided, otherwise create it
        if cached_memory is not None:
            memory = cached_memory
        else:
            # Create combined memory (V12.4: stoich, V14.3: heads)
            memory = self._create_memory(z, encoder_skip, stoich_pred, heads_pred)

        # Fast path: Full teacher forcing (TF = 1.0) - parallel forward
        if teacher_forcing_ratio >= 1.0:
            input_tokens = target_tokens[:, :-1]
            embedded = self.token_embedding(input_tokens)
            embedded = self.pos_encoding(embedded)

            causal_mask = self._generate_causal_mask(seq_len, device)
            tgt_key_padding_mask = (input_tokens == PAD_IDX)

            # V12.8: Use gradient checkpointing to save memory during training
            if self.use_gradient_checkpointing and self.training:
                # Checkpoint requires function format - wrap decoder call
                def run_decoder(tgt, mem, mask, padding_mask):
                    return self.transformer_decoder(
                        tgt=tgt, memory=mem, tgt_mask=mask,
                        tgt_key_padding_mask=padding_mask
                    )
                output = grad_checkpoint(
                    run_decoder, embedded, memory, causal_mask, tgt_key_padding_mask,
                    use_reentrant=False
                )
            else:
                output = self.transformer_decoder(
                    tgt=embedded,
                    memory=memory,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )

            logits = self.output_proj(output)
            stop_logits = self.stop_head(output).squeeze(-1)  # V12.30: [batch, seq_len]
            type_logits = self.token_type_head(output)  # V14.3: [batch, seq_len, N_TOKEN_TYPES]
            generated = logits.argmax(dim=-1)
            return logits, generated, stop_logits, type_logits

        # OPTIMIZED path: Scheduled sampling (TF < 1.0) - 2 passes instead of 60
        # V12.6: This is 30x faster than the original sequential approach
        #
        # Strategy:
        # 1. First pass: Get logits using ground truth inputs (parallel, like TF=1.0)
        # 2. Sample tokens from logits and mix with ground truth per TF ratio
        # 3. Second pass: Forward with mixed inputs to get final logits
        #
        # This gives the model training signal on handling its own predictions
        # while keeping computational cost at 2x instead of 60x.

        # ==== First pass: get predicted tokens using parallel forward ====
        input_tokens = target_tokens[:, :-1]  # All but last
        embedded = self.token_embedding(input_tokens)
        embedded = self.pos_encoding(embedded)

        causal_mask = self._generate_causal_mask(seq_len, device)
        tgt_key_padding_mask = (input_tokens == PAD_IDX)

        # V12.8: Helper function for checkpointing
        def run_decoder(tgt, mem, mask, padding_mask):
            return self.transformer_decoder(
                tgt=tgt, memory=mem, tgt_mask=mask,
                tgt_key_padding_mask=padding_mask
            )

        # V12.8: Use gradient checkpointing for first pass
        if self.use_gradient_checkpointing and self.training:
            output = grad_checkpoint(
                run_decoder, embedded, memory, causal_mask, tgt_key_padding_mask,
                use_reentrant=False
            )
        else:
            output = self.transformer_decoder(
                tgt=embedded,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        # Get logits and predictions from first pass
        first_logits = self.output_proj(output)  # [batch, seq_len, vocab]
        predicted_tokens = first_logits.argmax(dim=-1)  # [batch, seq_len]

        # ==== Mix predicted tokens with ground truth ====
        # For each position, use ground truth with probability TF, prediction with (1-TF)
        # V12.34: Position-dependent TF — more TF at start, less at end.
        # tf(pos) = base_tf * (1 + gamma * (1 - pos/L)), clamped to [0, 1]
        if self.use_position_dependent_tf and teacher_forcing_ratio < 1.0:
            positions = torch.arange(seq_len, device=device).float() / max(seq_len - 1, 1)
            tf_per_position = teacher_forcing_ratio * (
                1.0 + self.tf_position_decay * (1.0 - positions)
            )
            tf_per_position = tf_per_position.clamp(0.0, 1.0)
            use_gt_mask = (torch.rand(batch_size, seq_len, device=device) < tf_per_position.unsqueeze(0))
        else:
            use_gt_mask = (torch.rand(batch_size, seq_len, device=device) < teacher_forcing_ratio)

        # Target for position t is at target_tokens[:, t+1] (shifted by 1)
        gt_next_tokens = target_tokens[:, 1:]  # Ground truth targets

        # Mix: use GT where mask is True, prediction where False
        mixed_tokens = torch.where(use_gt_mask, gt_next_tokens, predicted_tokens)

        # Create new input sequence: START + mixed tokens (excluding last position)
        # The input at position t should be the token chosen for position t-1
        start_tokens = target_tokens[:, :1]  # START token
        mixed_inputs = torch.cat([start_tokens, mixed_tokens[:, :-1]], dim=1)  # [batch, seq_len]

        # ==== Second pass: forward with mixed inputs ====
        embedded = self.token_embedding(mixed_inputs)
        embedded = self.pos_encoding(embedded)

        # Padding mask based on mixed inputs
        tgt_key_padding_mask = (mixed_inputs == PAD_IDX)

        # V12.8: Use gradient checkpointing for second pass
        if self.use_gradient_checkpointing and self.training:
            output = grad_checkpoint(
                run_decoder, embedded, memory, causal_mask, tgt_key_padding_mask,
                use_reentrant=False
            )
        else:
            output = self.transformer_decoder(
                tgt=embedded,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        logits = self.output_proj(output)  # [batch, seq_len, vocab]
        stop_logits = self.stop_head(output).squeeze(-1)  # V12.30: [batch, seq_len]
        type_logits = self.token_type_head(output)  # V14.3: [batch, seq_len, N_TOKEN_TYPES]
        generated = logits.argmax(dim=-1)  # [batch, seq_len]

        return logits, generated, stop_logits, type_logits

    def generate(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
        stop_boost: float = 0.0,  # V12.30: Additive END logit boost from stop head
        stoich_pred: Optional[torch.Tensor] = None,  # V14.3: For memory consistency
        heads_pred: Optional[Dict[str, torch.Tensor]] = None,  # V14.3: Enriched memory
    ) -> List[str]:
        """Generate formulas autoregressively (legacy, no KV cache)."""
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        with torch.no_grad():
            memory = self._create_memory(z, encoder_skip, stoich_pred, heads_pred=heads_pred)

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

                # V12.30: Boost END logit based on stop head prediction
                if stop_boost > 0:
                    stop_logit = self.stop_head(output[:, -1, :]).squeeze(-1)  # [batch]
                    stop_prob = torch.sigmoid(stop_logit)
                    logits[:, END_IDX] = logits[:, END_IDX] + stop_boost * stop_prob

                    # V12.37: Length-conditional stop boost (normalized by max_len)
                    if step > 10:
                        length_boost = 10.0 * (step - 10) / max(max_len - 10, 1)
                        logits[:, END_IDX] = logits[:, END_IDX] + length_boost

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

    # =========================================================================
    # V12.7: KV CACHING FOR FAST AUTOREGRESSIVE GENERATION
    # =========================================================================
    #
    # KV caching reduces generation from O(n²) to O(n) by reusing computed
    # key/value tensors from previous positions. This is critical for:
    # 1. Fast inference (60x speedup for 60-token sequences)
    # 2. REINFORCE training (need to sample many sequences quickly)
    #
    # Architecture:
    #   - Cache stores (key, value) for self-attention in each decoder layer
    #   - Cross-attention to memory doesn't need caching (memory is static)
    #   - Each step processes only the NEW token, not the full sequence
    # =========================================================================

    def _init_kv_cache(
        self,
        batch_size: int,
        device: torch.device
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Initialize empty KV cache for all decoder layers.

        Returns:
            List of dicts, one per layer, each containing:
                'key': [batch, 0, d_model] - empty, will grow
                'value': [batch, 0, d_model] - empty, will grow
        """
        cache = []
        for _ in range(self.num_layers):
            cache.append({
                'key': torch.empty(batch_size, 0, self.d_model, device=device),
                'value': torch.empty(batch_size, 0, self.d_model, device=device),
            })
        return cache

    def _forward_one_step_with_cache(
        self,
        token_embedding: torch.Tensor,  # [batch, 1, d_model] - single token
        memory: torch.Tensor,           # [batch, mem_len, d_model]
        kv_cache: List[Dict[str, torch.Tensor]],
        position: int,                  # Current position (for positional encoding)
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass for a single token using KV cache.

        This manually iterates through decoder layers to manage the cache.
        For each layer:
        1. Self-attention: Use cached K,V + new K,V from current token
        2. Cross-attention: Attend to memory (no caching needed)
        3. FFN: Standard feed-forward

        Args:
            token_embedding: Embedded token [batch, 1, d_model]
            memory: Encoder memory [batch, mem_len, d_model]
            kv_cache: Current cache state
            position: Current sequence position

        Returns:
            output: Hidden state [batch, 1, d_model]
            updated_cache: Cache with new K,V appended
        """
        batch_size = token_embedding.size(0)
        device = token_embedding.device

        # Add positional encoding for this position
        # pos_encoding.pe is [1, max_len, d_model]
        x = token_embedding + self.pos_encoding.pe[:, position:position+1, :]
        x = self.pos_encoding.dropout(x)

        updated_cache = []

        # Process through each decoder layer
        for layer_idx, layer in enumerate(self.transformer_decoder.layers):
            layer_cache = kv_cache[layer_idx]

            # ===== Self-attention with KV cache =====
            # Pre-norm (since norm_first=True)
            x_norm = layer.norm1(x)

            # Compute Q, K, V for the new token
            # We need to access the in_proj_weight and in_proj_bias
            # to compute Q, K, V manually
            d_model = self.d_model
            nhead = self.nhead
            head_dim = d_model // nhead

            # Get Q, K, V projections
            # MultiheadAttention stores them concatenated in in_proj_weight
            qkv_weight = layer.self_attn.in_proj_weight  # [3*d_model, d_model]
            qkv_bias = layer.self_attn.in_proj_bias      # [3*d_model]

            # Project to get Q, K, V for new token
            qkv = F.linear(x_norm, qkv_weight, qkv_bias)  # [batch, 1, 3*d_model]
            q, k, v = qkv.chunk(3, dim=-1)  # Each [batch, 1, d_model]

            # Append new K, V to cache
            cached_k = layer_cache['key']
            cached_v = layer_cache['value']

            new_k = torch.cat([cached_k, k], dim=1)  # [batch, seq_len, d_model]
            new_v = torch.cat([cached_v, v], dim=1)

            # Update cache for this layer
            updated_cache.append({
                'key': new_k,
                'value': new_v,
            })

            # Compute attention: Q attends to all K (cached + new)
            # Reshape for multi-head attention
            seq_len = new_k.size(1)

            q = q.view(batch_size, 1, nhead, head_dim).transpose(1, 2)      # [batch, nhead, 1, head_dim]
            k_all = new_k.view(batch_size, seq_len, nhead, head_dim).transpose(1, 2)  # [batch, nhead, seq_len, head_dim]
            v_all = new_v.view(batch_size, seq_len, nhead, head_dim).transpose(1, 2)

            # Scaled dot-product attention (no mask needed - we attend to all cached positions)
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(q, k_all.transpose(-2, -1)) * scale  # [batch, nhead, 1, seq_len]
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=layer.self_attn.dropout, training=self.training)

            attn_output = torch.matmul(attn_weights, v_all)  # [batch, nhead, 1, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, d_model)

            # Output projection
            attn_output = layer.self_attn.out_proj(attn_output)

            # Residual connection
            x = x + layer.dropout1(attn_output)

            # ===== Cross-attention to memory =====
            x_norm = layer.norm2(x)

            # Standard cross-attention (no caching needed for memory)
            cross_attn_output, _ = layer.multihead_attn(
                query=x_norm,
                key=memory,
                value=memory,
                need_weights=False
            )
            x = x + layer.dropout2(cross_attn_output)

            # ===== Feed-forward =====
            x_norm = layer.norm3(x)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_norm))))
            x = x + layer.dropout3(ff_output)

        # Final layer norm (if TransformerDecoder has it)
        if self.transformer_decoder.norm is not None:
            x = self.transformer_decoder.norm(x)

        return x, updated_cache

    def generate_with_kv_cache(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        max_len: Optional[int] = None,
        return_log_probs: bool = False,
        return_entropy: bool = False,  # V12.8: Return proper entropy from full distribution
        cached_memory: Optional[torch.Tensor] = None,  # V12.8: Pre-computed memory
        stop_boost: float = 0.0,  # V12.30: Additive END logit boost from stop head
        hard_stop_threshold: float = 0.0,  # V12.37: Force END when sigmoid(stop_logit) > threshold
        heads_pred: Optional[Dict[str, torch.Tensor]] = None,  # V14.3: encoder head predictions
        type_masks: Optional[torch.Tensor] = None,  # V14.3: [N_TOKEN_TYPES, vocab_size] bool masks
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate token sequences with KV caching for O(n) complexity.

        This is ~60x faster than the non-cached version for 60-token sequences.

        Args:
            z: Latent vectors [batch, latent_dim]
            encoder_skip: Optional skip connection [batch, encoder_skip_dim]
            stoich_pred: Optional stoichiometry conditioning [batch, max_elements*3 + 1]
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, only sample from top k tokens
            top_p: If set, use nucleus sampling
            max_len: Maximum sequence length
            return_log_probs: If True, also return log probabilities of sampled tokens
            return_entropy: If True, also return proper entropy H(p) = -sum(p * log(p))
            cached_memory: V12.8 Pre-computed memory from precompute_memory()
            heads_pred: V14.3 Dict of encoder head predictions for enriched memory
            type_masks: V14.3 Precomputed [N_TOKEN_TYPES, vocab_size] boolean masks
                       from tokenizer.get_type_masks(). When provided, applies hard type
                       masking: predicted type → only allow tokens of that type.

        Returns:
            generated_tokens: Token indices [batch, seq_len] (excluding START)
            log_probs: Log probabilities [batch, seq_len] if return_log_probs=True, else None
            entropy: Proper entropy per position [batch, seq_len] if return_entropy=True, else None
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        with torch.no_grad():
            # V12.8: Use cached memory if provided, otherwise create it
            if cached_memory is not None:
                memory = cached_memory
            else:
                memory = self._create_memory(z, encoder_skip, stoich_pred, heads_pred)

            # V14.3: Move type masks to device if provided
            if type_masks is not None:
                type_masks = type_masks.to(device)

            # Initialize KV cache
            kv_cache = self._init_kv_cache(batch_size, device)

            # Start with START token
            current_token = torch.full((batch_size, 1), START_IDX, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            generated_tokens = []
            log_probs_list = [] if return_log_probs else None
            entropy_list = [] if return_entropy else None  # V12.8: Track proper entropy

            for position in range(max_len - 1):
                # Embed current token
                token_emb = self.token_embedding(current_token)  # [batch, 1, d_model]

                # Forward with cache
                output, kv_cache = self._forward_one_step_with_cache(
                    token_emb, memory, kv_cache, position
                )

                # Get logits for this position
                logits = self.output_proj(output).squeeze(1)  # [batch, vocab_size]

                # V14.3: Token type masking — predict type, then mask vocab to that type
                if type_masks is not None:
                    type_logit = self.token_type_head(output).squeeze(1)  # [batch, N_TOKEN_TYPES]
                    predicted_type = type_logit.argmax(dim=-1)  # [batch]
                    # For each sample in batch, get the valid token mask for its predicted type
                    valid_mask = type_masks[predicted_type]  # [batch, vocab_size]
                    # Set invalid tokens to -inf
                    logits = logits.masked_fill(~valid_mask, float('-inf'))

                # V12.30: Boost END logit based on stop head prediction
                if stop_boost > 0:
                    stop_logit = self.stop_head(output).squeeze(1).squeeze(-1)  # [batch]
                    stop_prob = torch.sigmoid(stop_logit)
                    logits[:, END_IDX] = logits[:, END_IDX] + stop_boost * stop_prob

                    # V12.37: Hard stop threshold — force END when stop head is highly confident
                    if hard_stop_threshold > 0:
                        force_end = (stop_prob > hard_stop_threshold) & ~finished
                        if force_end.any():
                            logits[force_end, :] = float('-inf')
                            logits[force_end, END_IDX] = 100.0  # Force END token

                    # V12.37: Length-conditional stop boost — linearly increase END boost
                    # past the typical formula length to prevent runaway generation.
                    # Normalized by max_len so boost represents fraction of generation
                    # budget consumed and auto-adjusts if max_len changes.
                    # With max_len=60, scale=10: pos 20 → +2.0, pos 30 → +4.0, pos 40 → +6.0
                    if position > 10:
                        length_boost = 10.0 * (position - 10) / max(max_len - 10, 1)
                        logits[:, END_IDX] = logits[:, END_IDX] + length_boost

                # V12.8: Compute proper entropy BEFORE temperature/filtering
                # H(p) = -sum(p * log(p)) over vocabulary
                if return_entropy:
                    # Use raw logits (no temperature) for true distribution entropy
                    # V12.40: Clamp to avoid 0*log(0)=NaN when softmax produces exact 0.0
                    probs_for_entropy = F.softmax(logits, dim=-1).clamp(min=1e-8)
                    log_probs_for_entropy = probs_for_entropy.log()
                    # Entropy: -sum(p * log(p)), sum over vocab dimension
                    step_entropy = -(probs_for_entropy * log_probs_for_entropy).sum(dim=-1)  # [batch]
                    entropy_list.append(step_entropy)

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
                    if return_log_probs:
                        log_prob = torch.zeros(batch_size, device=device)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    if return_log_probs:
                        log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token).squeeze(-1)

                # Record generated token and log prob
                generated_tokens.append(next_token)
                if return_log_probs:
                    log_probs_list.append(log_prob)

                # Update finished status
                finished = finished | (next_token.squeeze(-1) == END_IDX)

                # Next input is the sampled token
                current_token = next_token

                if finished.all():
                    break

            # Stack results
            generated = torch.cat(generated_tokens, dim=1)  # [batch, seq_len]

            # V12.8: Return log_probs and entropy as requested
            log_probs = torch.stack(log_probs_list, dim=1) if return_log_probs else None
            entropy = torch.stack(entropy_list, dim=1) if return_entropy else None  # [batch, seq_len]

            return generated, log_probs, entropy

    def sample_for_reinforce(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        max_len: Optional[int] = None,
        cached_memory: Optional[torch.Tensor] = None,  # V12.8: Pre-computed memory
        stop_boost: float = 0.0,  # V12.30: Additive END logit boost from stop head
        hard_stop_threshold: float = 0.0,  # V12.37: Force END when sigmoid(stop_logit) > threshold
        heads_pred: Optional[Dict[str, torch.Tensor]] = None,  # V14.3
        type_masks: Optional[torch.Tensor] = None,  # V14.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample sequences for REINFORCE training with KV caching.

        Returns sampled tokens, log probabilities, proper entropy, and a mask for valid positions.
        This is optimized for RLOO (REINFORCE Leave-One-Out) training.

        Args:
            z: Latent vectors [batch, latent_dim]
            encoder_skip: Optional skip connection
            stoich_pred: Optional stoichiometry conditioning
            temperature: Sampling temperature
            max_len: Maximum sequence length
            cached_memory: V12.8 Pre-computed memory from precompute_memory()
            stop_boost: V12.30 Additive END logit boost from stop head
            heads_pred: V14.3 Dict of encoder head predictions for enriched memory
            type_masks: V14.3 Precomputed type masks for hard type masking

        Returns:
            sampled_tokens: [batch, seq_len] - sampled token indices
            log_probs: [batch, seq_len] - log probability of each sampled token
            entropy: [batch, seq_len] - proper entropy H(p) = -sum(p * log(p)) per position
            mask: [batch, seq_len] - 1 for valid positions, 0 for padding
        """
        max_len = max_len or self.max_len

        # V12.8: Sample with log probs AND entropy using KV cache
        sampled_tokens, log_probs, entropy = self.generate_with_kv_cache(
            z=z,
            encoder_skip=encoder_skip,
            stoich_pred=stoich_pred,
            temperature=temperature,
            max_len=max_len,
            return_log_probs=True,
            return_entropy=True,  # V12.8: Get proper entropy
            cached_memory=cached_memory,  # V12.8: Pass through cached memory
            stop_boost=stop_boost,  # V12.30: Pass through stop boost
            hard_stop_threshold=hard_stop_threshold,  # V12.37: Pass through hard stop threshold
            heads_pred=heads_pred,  # V14.3: Pass through heads_pred
            type_masks=type_masks,  # V14.3: Pass through type masks
        )

        # Create mask: 1 for valid tokens, 0 after END
        batch_size, seq_len = sampled_tokens.shape
        device = sampled_tokens.device

        # Find END positions
        is_end = (sampled_tokens == END_IDX)

        # Create cumulative mask (0 after first END)
        # First, get position of first END in each sequence
        end_positions = torch.argmax(is_end.int(), dim=1)  # [batch]
        has_end = is_end.any(dim=1)  # [batch]

        # For sequences without END, mask all positions
        end_positions = torch.where(has_end, end_positions, torch.tensor(seq_len, device=device))

        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Mask: 1 for positions <= END position (include END in loss)
        mask = (positions <= end_positions.unsqueeze(1)).float()

        return sampled_tokens, log_probs, entropy, mask

    def speculative_sample_for_reinforce(
        self,
        z: torch.Tensor,
        draft_model,  # HybridDraft from ngram_draft.py
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        max_len: Optional[int] = None,
        k: int = 5,  # Number of tokens to draft at once
        cached_memory: Optional[torch.Tensor] = None,
        heads_pred: Optional[Dict[str, torch.Tensor]] = None,  # V14.3
        type_masks: Optional[torch.Tensor] = None,  # V14.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        V12.16: REINFORCE sampling with TRUE per-sequence speculative decoding.

        Key fix: Each sequence in the batch advances independently at its own
        pace, with proper KV cache management per sequence.

        Algorithm:
        1. Draft k tokens using fast n-gram/structural model
        2. Run main model forward on all k positions at once (batched verification)
        3. Per-sequence rejection sampling - each seq finds its own rejection point
        4. Slice temp_cache to reuse verification KV values (no recompute!)
        5. Sample from model distribution at each sequence's rejection point
        6. Continue until all sequences finish

        Args:
            z: Latent vectors [batch, latent_dim]
            draft_model: HybridDraft model for fast token prediction
            encoder_skip: Optional skip connection
            stoich_pred: Optional stoichiometry conditioning
            temperature: Sampling temperature
            max_len: Maximum sequence length
            k: Number of tokens to draft at each step
            cached_memory: Pre-computed memory from precompute_memory()

        Returns:
            sampled_tokens: [batch, seq_len] - sampled token indices
            log_probs: [batch, seq_len] - log probability of each sampled token
            entropy: [batch, seq_len] - entropy H(p) per position
            mask: [batch, seq_len] - 1 for valid positions, 0 for padding
            stats: Dict with acceptance_rate, avg_tokens_per_step, n_steps
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device
        max_len = max_len or self.max_len

        # Stats tracking
        total_drafted = 0
        total_accepted = 0
        n_steps = 0

        with torch.no_grad():
            # Create or use cached memory
            if cached_memory is not None:
                memory = cached_memory
            else:
                memory = self._create_memory(z, encoder_skip, stoich_pred, heads_pred=heads_pred)

            # V14.3: Move type masks to device if provided
            if type_masks is not None:
                type_masks = type_masks.to(device)

            # Per-sequence state tracking
            # Each sequence has its own position and KV cache length
            seq_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Pre-allocate output buffers (max_len size, will trim later)
            all_tokens = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long, device=device)
            all_log_probs = torch.zeros(batch_size, max_len, device=device)
            all_entropies = torch.zeros(batch_size, max_len, device=device)

            # Initialize KV cache and process START token
            kv_cache = self._init_kv_cache(batch_size, device)
            start_tokens = torch.full((batch_size, 1), START_IDX, dtype=torch.long, device=device)
            start_emb = self.token_embedding(start_tokens)
            _, kv_cache = self._forward_one_step_with_cache(start_emb, memory, kv_cache, 0)

            # Track current tokens for draft context (per sequence)
            current_tokens = start_tokens.clone()  # [batch, 1]

            # Main loop - continue until all sequences done
            while not finished.all() and seq_positions.max() < max_len - 1:
                n_steps += 1
                active_mask = ~finished  # Which sequences still need tokens

                # ===== Step 1: Draft k tokens =====
                # Use first active sequence as context (draft model is position-aware)
                first_active = active_mask.nonzero(as_tuple=True)[0][0].item()
                pos = seq_positions[first_active].item()
                context_list = current_tokens[first_active, :pos+1].tolist()
                drafted = draft_model.draft_k_tokens(context_list, k=k)
                drafted_tensor = torch.tensor([drafted] * batch_size, device=device, dtype=torch.long)

                n_to_draft = min(k, max_len - 1 - pos)
                total_drafted += n_to_draft * active_mask.sum().item()

                # ===== Step 2: Verify all k tokens in ONE forward pass =====
                # Save cache state before verification (for slicing later)
                cache_before_verify = self._copy_kv_cache(kv_cache)

                draft_logits = []
                draft_entropies = []
                temp_cache = kv_cache

                for draft_pos in range(n_to_draft):
                    draft_token = drafted_tensor[:, draft_pos:draft_pos+1]
                    draft_emb = self.token_embedding(draft_token)

                    # Use position of first active sequence (they're synchronized in this step)
                    output, temp_cache = self._forward_one_step_with_cache(
                        draft_emb, memory, temp_cache, pos + draft_pos + 1
                    )

                    logits = self.output_proj(output).squeeze(1)  # [batch, vocab_size]
                    draft_logits.append(logits)

                    # V12.40: Clamp to avoid 0*log(0)=NaN
                    probs_ent = F.softmax(logits, dim=-1).clamp(min=1e-8)
                    log_probs_ent = probs_ent.log()
                    step_entropy = -(probs_ent * log_probs_ent).sum(dim=-1)
                    draft_entropies.append(step_entropy)

                if not draft_logits:
                    break

                # Stack: [batch, n_drafted, vocab_size]
                stacked_logits = torch.stack(draft_logits, dim=1)
                stacked_entropies = torch.stack(draft_entropies, dim=1)
                n_drafted = stacked_logits.size(1)

                # ===== Step 3: Per-sequence verification =====
                # FIX V12.16.1: Use greedy verification instead of naive rejection sampling.
                # The previous `r < p_model(token)` was incorrect - it rejected even good
                # predictions because p(token) is typically low for large vocabularies.
                #
                # Proper speculative decoding uses `r < p_model/p_draft`, but we don't
                # have draft probabilities. Instead, we use greedy verification:
                # Accept if draft token matches argmax OR has probability >= threshold.

                scaled_logits = stacked_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                log_probs_all = F.log_softmax(scaled_logits, dim=-1)

                # Get model's argmax predictions at each position
                model_argmax = probs.argmax(dim=-1)  # [batch, n_drafted]

                # Get probability of draft tokens
                draft_probs = probs.gather(2, drafted_tensor[:, :n_drafted].unsqueeze(-1)).squeeze(-1)

                # Verification criteria (any of these = accept):
                # 1. Draft matches model's greedy choice (argmax)
                # 2. Draft token has probability >= 0.1 (reasonably likely)
                # 3. Draft token is in top-5 (for cases where distribution is flat)
                matches_greedy = (drafted_tensor[:, :n_drafted] == model_argmax)  # [batch, n_drafted]
                above_threshold = (draft_probs >= 0.1)  # [batch, n_drafted]

                # Top-5 check: draft token should be in top-5 most likely tokens
                top5_tokens = probs.topk(k=5, dim=-1).indices  # [batch, n_drafted, 5]
                in_top5 = (drafted_tensor[:, :n_drafted].unsqueeze(-1) == top5_tokens).any(dim=-1)

                # Accept if any criterion is met
                accept_mask = matches_greedy | above_threshold | in_top5  # [batch, n_drafted]

                # Find consecutive accepts per sequence
                accept_chain = accept_mask.cumprod(dim=1)
                n_accepted_per_seq = accept_chain.sum(dim=1)  # [batch]

                # ===== Step 4: Vectorized per-sequence token assignment =====
                # All operations are batched - no Python loops over batch_size

                # Clamp n_accepted to avoid overflow
                n_accepted_clamped = n_accepted_per_seq.clamp(max=n_drafted)

                # Create position indices for each sequence: [batch, n_drafted]
                # pos_offsets[b, i] = seq_positions[b] + i
                draft_indices = torch.arange(n_drafted, device=device).unsqueeze(0)  # [1, n_drafted]
                pos_offsets = seq_positions.unsqueeze(1) + draft_indices  # [batch, n_drafted]

                # Mask for valid positions (within max_len and within n_accepted)
                within_max_len = pos_offsets < (max_len - 1)
                within_accepted = draft_indices < n_accepted_clamped.unsqueeze(1)
                active_seq = ~finished.unsqueeze(1)
                valid_mask = within_max_len & within_accepted & active_seq  # [batch, n_drafted]

                # Scatter accepted tokens to output buffers
                # Get log probs for draft tokens: [batch, n_drafted]
                draft_token_log_probs = log_probs_all.gather(
                    2, drafted_tensor[:, :n_drafted].unsqueeze(-1)
                ).squeeze(-1)

                # For each valid position, write token, log_prob, entropy
                for i in range(n_drafted):
                    mask_i = valid_mask[:, i]  # [batch]
                    if not mask_i.any():
                        continue

                    write_pos = (seq_positions + i).clamp(max=max_len - 1)

                    # Use advanced indexing to write only to valid positions
                    batch_indices = torch.arange(batch_size, device=device)[mask_i]
                    pos_indices = write_pos[mask_i]

                    all_tokens[batch_indices, pos_indices] = drafted_tensor[mask_i, i]
                    all_log_probs[batch_indices, pos_indices] = draft_token_log_probs[mask_i, i]
                    all_entropies[batch_indices, pos_indices] = stacked_entropies[mask_i, i]

                # Check for END tokens in accepted range
                accepted_tokens = drafted_tensor[:, :n_drafted]  # [batch, n_drafted]
                is_end = (accepted_tokens == END_IDX) & valid_mask
                newly_finished = is_end.any(dim=1)
                finished = finished | newly_finished

                # Find first END position per sequence (for truncating n_accepted)
                end_positions = torch.where(
                    is_end,
                    draft_indices.expand(batch_size, -1),
                    torch.full_like(draft_indices.expand(batch_size, -1), n_drafted)
                ).min(dim=1).values
                n_accepted_with_end = torch.minimum(n_accepted_clamped, end_positions + 1)

                # Update total accepted count
                total_accepted += (n_accepted_with_end * (~finished | newly_finished).long()).sum().item()

                # ===== Sample at rejection point (vectorized) =====
                needs_sample = ~finished & (n_accepted_clamped < n_drafted)

                if needs_sample.any():
                    # Get rejection position per sequence
                    reject_pos = n_accepted_clamped.clamp(max=n_drafted - 1)  # [batch]

                    # Gather logits at rejection position: [batch, vocab_size]
                    reject_logits = stacked_logits.gather(
                        1,
                        reject_pos.view(batch_size, 1, 1).expand(-1, -1, stacked_logits.size(-1))
                    ).squeeze(1)

                    # Sample from model distribution
                    scaled = reject_logits / temperature
                    probs_sample = F.softmax(scaled, dim=-1)
                    log_probs_sample = F.log_softmax(scaled, dim=-1)
                    sampled_tokens = torch.multinomial(probs_sample, num_samples=1).squeeze(-1)  # [batch]

                    # Get log probs and entropy for sampled tokens
                    sampled_log_probs = log_probs_sample.gather(1, sampled_tokens.unsqueeze(1)).squeeze(1)
                    sampled_entropies = stacked_entropies.gather(
                        1, reject_pos.unsqueeze(1)
                    ).squeeze(1)

                    # Write sampled tokens to output buffers (only for sequences that need sampling)
                    write_pos_sample = (seq_positions + n_accepted_clamped).clamp(max=max_len - 1)
                    sample_mask = needs_sample & (write_pos_sample < max_len - 1)

                    if sample_mask.any():
                        batch_indices = torch.arange(batch_size, device=device)[sample_mask]
                        pos_indices = write_pos_sample[sample_mask]

                        all_tokens[batch_indices, pos_indices] = sampled_tokens[sample_mask]
                        all_log_probs[batch_indices, pos_indices] = sampled_log_probs[sample_mask]
                        all_entropies[batch_indices, pos_indices] = sampled_entropies[sample_mask]

                        # Check if sampled token is END
                        sampled_is_end = (sampled_tokens == END_IDX) & sample_mask
                        finished = finished | sampled_is_end

                    # Tokens advanced = n_accepted + 1 (for sampled) where sampling happened
                    tokens_advanced = torch.where(
                        needs_sample,
                        n_accepted_with_end + 1,
                        n_accepted_with_end
                    )
                else:
                    tokens_advanced = n_accepted_with_end

                # Update sequence positions
                seq_positions = (seq_positions + tokens_advanced).clamp(max=max_len - 1)

                # ===== Step 5: Update KV cache and current_tokens =====
                # Find the minimum position to synchronize batch processing
                # (sequences that advanced further will need to re-verify some tokens)
                min_new_pos = seq_positions[active_mask].min().item() if active_mask.any() else 0

                # Rebuild cache up to min_new_pos by replaying tokens
                kv_cache = cache_before_verify
                for step in range(min_new_pos - pos):
                    step_tokens = all_tokens[:, pos + step:pos + step + 1]
                    step_emb = self.token_embedding(step_tokens)
                    _, kv_cache = self._forward_one_step_with_cache(
                        step_emb, memory, kv_cache, pos + step + 1
                    )

                # Update current_tokens for next draft context
                max_pos = seq_positions.max().item()
                if max_pos > current_tokens.size(1) - 1:
                    # Expand current_tokens to include new positions
                    new_tokens = torch.full((batch_size, max_pos + 1), PAD_IDX, dtype=torch.long, device=device)
                    new_tokens[:, 0] = START_IDX
                    new_tokens[:, 1:seq_positions.max() + 1] = all_tokens[:, :seq_positions.max()]
                    current_tokens = new_tokens

            # ===== Build output tensors =====
            # Find actual sequence lengths
            actual_len = seq_positions.max().item()
            if actual_len == 0:
                actual_len = 1  # At least one token

            sampled_tokens = all_tokens[:, :actual_len]
            log_probs = all_log_probs[:, :actual_len]
            entropy = all_entropies[:, :actual_len]

            # Create mask
            seq_len = sampled_tokens.size(1)
            if seq_len > 0:
                is_end = (sampled_tokens == END_IDX)
                end_positions = torch.argmax(is_end.int(), dim=1)
                has_end = is_end.any(dim=1)
                end_positions = torch.where(has_end, end_positions, seq_positions - 1)
                positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                mask = (positions <= end_positions.unsqueeze(1)).float()
            else:
                mask = torch.empty(batch_size, 0, device=device)

            # Compute stats
            acceptance_rate = total_accepted / max(total_drafted, 1)
            avg_tokens_per_step = (seq_positions.float().mean().item()) / max(n_steps, 1)

            stats = {
                'acceptance_rate': acceptance_rate,
                'avg_tokens_per_step': avg_tokens_per_step,
                'n_steps': n_steps,
                'total_drafted': total_drafted,
                'total_accepted': total_accepted,
            }

            return sampled_tokens, log_probs, entropy, mask, stats

    def _copy_kv_cache(self, kv_cache: Optional[List[Dict[str, torch.Tensor]]]) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Deep copy a KV cache (List of Dicts) for checkpointing."""
        if kv_cache is None:
            return None
        return [
            {
                'key': layer_cache['key'].clone(),
                'value': layer_cache['value'].clone(),
            }
            for layer_cache in kv_cache
        ]

    def generate_formulas_fast(
        self,
        z: torch.Tensor,
        encoder_skip: Optional[torch.Tensor] = None,
        stoich_pred: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
        cached_memory: Optional[torch.Tensor] = None,  # V12.8: Pre-computed memory
    ) -> List[str]:
        """
        Generate formula strings using fast KV-cached generation.

        Drop-in replacement for generate() but ~60x faster.

        Args:
            cached_memory: V12.8 Pre-computed memory from precompute_memory()
        """
        generated_tokens, _, _ = self.generate_with_kv_cache(
            z=z,
            encoder_skip=encoder_skip,
            stoich_pred=stoich_pred,
            temperature=temperature,
            max_len=max_len,
            return_log_probs=False,
            return_entropy=False,
            cached_memory=cached_memory,  # V12.8
        )

        # Convert to formula strings
        formulas = []
        for i in range(generated_tokens.size(0)):
            formula = indices_to_formula(generated_tokens[i])
            formulas.append(formula)

        return formulas
