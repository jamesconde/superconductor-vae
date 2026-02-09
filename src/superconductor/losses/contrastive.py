"""
Supervised Contrastive Loss for SC vs non-SC latent space separation.

Based on Khosla et al. (2020) "Supervised Contrastive Learning" (SupCon)
with adaptations for superconductor materials science.

Used to train the VAE encoder to push SC and non-SC latent representations apart
while clustering SC families together. See docs/CONTRASTIVE_LEARNING_DESIGN.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperconductorContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for SC vs non-SC separation.

    Projects latent vectors onto unit hypersphere and uses cosine similarity
    with temperature scaling. Positives = same class, negatives = different class.

    Args:
        temperature: Temperature for scaling similarity (0.07 = standard).
        base_temperature: Base temperature for gradient scaling.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            z: [batch, latent_dim] - latent representations from encoder.
            labels: [batch] - integer class labels.
                    Simple: 0=non-SC, 1=SC
                    Extended: 0-N=SC families, N+1=non-SC categories

        Returns:
            Scalar loss value.
        """
        device = z.device
        batch_size = z.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Project to unit hypersphere (cosine similarity)
        z_norm = F.normalize(z, dim=1)

        # Pairwise cosine similarity scaled by temperature
        sim = torch.mm(z_norm, z_norm.t()) / self.temperature  # [batch, batch]

        # Positive mask: same class, excluding self-pairs
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        # All-pairs mask excluding self
        logits_mask = torch.ones(batch_size, batch_size, device=device)
        logits_mask.fill_diagonal_(0)

        # Numerical stability: subtract max per row
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log-softmax over all non-self pairs
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average log-probability over positive pairs
        pos_count = pos_mask.sum(dim=1)
        # Avoid division by zero for samples with no positives in batch
        safe_pos_count = torch.clamp(pos_count, min=1.0)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / safe_pos_count

        # Zero out loss for samples with no positives (shouldn't happen with balanced batching)
        mean_log_prob = mean_log_prob * (pos_count > 0).float()

        # Scale by temperature ratio
        loss = -(self.temperature / self.base_temperature) * mean_log_prob

        return loss.mean()


# Category label mapping for extended contrastive labels
SC_CATEGORY_LABELS = {
    # Superconductor families (labels 0-7)
    'Cuprates': 0,
    'Iron-based': 1,
    'Bismuthates': 2,
    'Borocarbides': 3,
    'Elemental Superconductors': 4,
    'Hydrogen-rich Superconductors': 5,
    'Organic Superconductors': 6,
    'Other': 7,
    # Non-superconductor categories (labels 8-11)
    'Non-SC: Materials Project': 8,
    'Non-SC: Magnetic': 9,
    'Non-SC: Thermoelectric': 10,
    'Non-SC: Anisotropy': 11,
    # V12.19: High-pressure non-hydride SC (elemental HP, fullerene, nickelate, etc.)
    # Hydride HP-SC stay in class 5 (Hydrogen-rich). This class covers OTHER HP-SC
    # that are currently lumped into "Other" or their parent category.
    'High-pressure (non-hydride)': 12,
}

# Binary label: 1=SC, 0=non-SC
SC_BINARY_LABEL = 1
NON_SC_BINARY_LABEL = 0


def category_to_label(category: str, use_extended: bool = True,
                      requires_high_pressure: int = 0) -> int:
    """Map a category string to an integer label for contrastive loss.

    Args:
        category: Category string from the CSV.
        use_extended: If True, use per-family labels. If False, binary SC/non-SC.
        requires_high_pressure: 1 if material requires high pressure, 0 otherwise.
            When 1 and category is NOT 'Hydrogen-rich Superconductors', overrides
            to class 12 ('High-pressure (non-hydride)').

    Returns:
        Integer label.
    """
    if not use_extended:
        return NON_SC_BINARY_LABEL if category.startswith('Non-SC') else SC_BINARY_LABEL

    # V12.19: HP non-hydride override — cluster non-hydride HP-SC together
    if (requires_high_pressure == 1
            and category != 'Hydrogen-rich Superconductors'
            and not category.startswith('Non-SC')):
        return SC_CATEGORY_LABELS['High-pressure (non-hydride)']

    if category in SC_CATEGORY_LABELS:
        return SC_CATEGORY_LABELS[category]

    # Fallback: SC family not in map → Other SC; Non-SC not in map → generic non-SC
    if category.startswith('Non-SC'):
        return max(SC_CATEGORY_LABELS.values())  # Last label
    return SC_CATEGORY_LABELS['Other']
