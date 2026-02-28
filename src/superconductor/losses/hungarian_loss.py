"""
V16.0: Hungarian Matching Loss for set-based formula prediction.

Uses scipy.optimize.linear_sum_assignment to find optimal assignment
between predicted slots and ground truth (element, fraction) pairs,
then computes differentiable losses on matched pairs.

Cost matrix combines element CE and fraction MSE. Loss components:
- Element CE (with no_object_weight downweighting for empty slots)
- Fraction MSE (only on real-element matches)
- Presence BCE (occupied/empty binary supervision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

# scipy imported lazily to avoid mandatory dependency
_linear_sum_assignment = None


def _get_linear_sum_assignment():
    """Lazy import of scipy.optimize.linear_sum_assignment."""
    global _linear_sum_assignment
    if _linear_sum_assignment is None:
        from scipy.optimize import linear_sum_assignment
        _linear_sum_assignment = linear_sum_assignment
    return _linear_sum_assignment


class HungarianMatchingLoss(nn.Module):
    """Hungarian matching loss for set-based formula prediction.

    Computes optimal bipartite matching between predicted slots and
    ground truth (element, fraction) pairs using the Hungarian algorithm,
    then applies differentiable losses on the matched pairs.

    Args:
        n_elements: Number of chemical elements (118)
        n_slots: Number of prediction slots (12)
        element_ce_weight: CE weight in cost matrix for matching
        fraction_mse_weight: MSE weight in cost matrix for matching
        presence_bce_weight: Presence BCE weight in total loss
        no_object_weight: Down-weight empty slot CE (0.1 = 10x less than real elements)
        fraction_loss_weight: Fraction MSE weight in total loss
        element_loss_weight: Element CE weight in total loss
    """

    def __init__(
        self,
        n_elements: int = 118,
        n_slots: int = 12,
        element_ce_weight: float = 1.0,
        fraction_mse_weight: float = 5.0,
        presence_bce_weight: float = 1.0,
        no_object_weight: float = 0.1,
        fraction_loss_weight: float = 5.0,
        element_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_slots = n_slots
        self.element_ce_weight = element_ce_weight
        self.fraction_mse_weight = fraction_mse_weight
        self.presence_bce_weight = presence_bce_weight
        self.no_object_weight = no_object_weight
        self.fraction_loss_weight = fraction_loss_weight
        self.element_loss_weight = element_loss_weight

    @torch.no_grad()
    def compute_matching(
        self,
        element_logits: torch.Tensor,
        fraction_pred: torch.Tensor,
        gt_elements: torch.Tensor,
        gt_fractions: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute optimal Hungarian matching for each sample in the batch.

        Args:
            element_logits: [B, n_slots, n_elements+1] — predicted element logits
            fraction_pred: [B, n_slots] — predicted fractions
            gt_elements: [B, max_elements] — ground truth atomic numbers (0=pad)
            gt_fractions: [B, max_elements] — ground truth fractions
            gt_mask: [B, max_elements] — True for present elements

        Returns:
            List of (row_indices, col_indices) tuples per sample.
            row_indices: matched slot indices
            col_indices: matched GT column indices (0..n_slots-1)
                Columns 0..n_gt-1 correspond to real GT elements
                Columns n_gt..n_slots-1 correspond to "no-object" padding
        """
        linear_sum_assignment = _get_linear_sum_assignment()
        B = element_logits.shape[0]

        # Log-softmax for CE cost
        log_probs = F.log_softmax(element_logits, dim=-1)  # [B, n_slots, n_elements+1]

        matchings = []
        for b in range(B):
            n_gt = gt_mask[b].sum().item()
            n_gt = int(n_gt)

            # Build n_slots × n_slots cost matrix
            cost = torch.zeros(self.n_slots, self.n_slots, device='cpu')

            if n_gt > 0:
                gt_elem_b = gt_elements[b, :n_gt].long()  # [n_gt] atomic numbers
                gt_frac_b = gt_fractions[b, :n_gt]          # [n_gt] fractions

                # Columns 0..n_gt-1: real GT elements
                for j in range(n_gt):
                    elem_idx = gt_elem_b[j].item()
                    # CE cost: negative log probability of correct element
                    ce_cost = -log_probs[b, :, elem_idx].cpu()  # [n_slots]
                    # MSE cost: squared difference in fraction
                    frac_cost = (fraction_pred[b].cpu() - gt_frac_b[j].cpu()) ** 2  # [n_slots]
                    cost[:, j] = (self.element_ce_weight * ce_cost +
                                  self.fraction_mse_weight * frac_cost)

            # Columns n_gt..n_slots-1: no-object padding
            no_obj_cost = -log_probs[b, :, 0].cpu()  # Cost of predicting class 0 (empty)
            for j in range(n_gt, self.n_slots):
                cost[:, j] = self.element_ce_weight * no_obj_cost

            # Solve Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost.numpy())
            matchings.append((
                torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long),
            ))

        return matchings

    def forward(
        self,
        element_logits: torch.Tensor,
        fraction_pred: torch.Tensor,
        presence_logits: torch.Tensor,
        gt_elements: torch.Tensor,
        gt_fractions: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute Hungarian matching loss.

        Args:
            element_logits: [B, n_slots, n_elements+1] from set decoder
            fraction_pred: [B, n_slots] from set decoder
            presence_logits: [B, n_slots] from set decoder
            gt_elements: [B, max_elements] ground truth atomic numbers
            gt_fractions: [B, max_elements] ground truth fractions
            gt_mask: [B, max_elements] True for present elements

        Returns:
            Dict with: total, element_loss, fraction_loss, presence_loss,
                       element_accuracy, fraction_mse, set_exact
        """
        B = element_logits.shape[0]
        device = element_logits.device

        # Compute optimal matching
        matchings = self.compute_matching(
            element_logits, fraction_pred, gt_elements, gt_fractions, gt_mask,
        )

        # Accumulators
        element_loss = torch.tensor(0.0, device=device)
        fraction_loss = torch.tensor(0.0, device=device)
        presence_loss = torch.tensor(0.0, device=device)
        total_elem_correct = 0
        total_elem_count = 0
        total_frac_mse = 0.0
        total_frac_count = 0
        total_set_exact = 0

        for b in range(B):
            row_ind, col_ind = matchings[b]
            row_ind = row_ind.to(device)
            col_ind = col_ind.to(device)
            n_gt = int(gt_mask[b].sum().item())

            # Separate real vs empty matches
            real_mask = col_ind < n_gt   # Matched to a real GT element
            empty_mask = ~real_mask       # Matched to no-object padding

            # --- Element CE loss ---
            # Build per-slot targets
            slot_targets = torch.zeros(self.n_slots, dtype=torch.long, device=device)
            if n_gt > 0 and real_mask.any():
                real_rows = row_ind[real_mask]
                real_cols = col_ind[real_mask]
                slot_targets[real_rows] = gt_elements[b, real_cols].long()

            # Empty slots target class 0
            if empty_mask.any():
                empty_rows = row_ind[empty_mask]
                slot_targets[empty_rows] = 0

            # Per-slot CE (no reduction)
            per_slot_ce = F.cross_entropy(
                element_logits[b], slot_targets, reduction='none',
            )  # [n_slots]

            # Weight: real elements at full weight, empty at no_object_weight
            ce_weights = torch.full((self.n_slots,), self.no_object_weight, device=device)
            if real_mask.any():
                ce_weights[row_ind[real_mask]] = 1.0

            element_loss = element_loss + (per_slot_ce * ce_weights).sum() / max(ce_weights.sum(), 1.0)

            # --- Fraction MSE loss (only on real-element matches) ---
            if n_gt > 0 and real_mask.any():
                real_rows = row_ind[real_mask]
                real_cols = col_ind[real_mask]
                pred_fracs = fraction_pred[b, real_rows]
                true_fracs = gt_fractions[b, real_cols].to(device)
                frac_mse = F.mse_loss(pred_fracs, true_fracs)
                fraction_loss = fraction_loss + frac_mse
                total_frac_mse += frac_mse.item() * real_mask.sum().item()
                total_frac_count += real_mask.sum().item()

            # --- Presence BCE loss ---
            presence_targets = torch.zeros(self.n_slots, device=device)
            if real_mask.any():
                presence_targets[row_ind[real_mask]] = 1.0
            presence_loss = presence_loss + F.binary_cross_entropy_with_logits(
                presence_logits[b], presence_targets,
            )

            # --- Metrics ---
            pred_elements = element_logits[b].argmax(dim=-1)  # [n_slots]
            total_elem_correct += (pred_elements == slot_targets).sum().item()
            total_elem_count += self.n_slots

            # Set exact: check if predicted element SET matches GT element SET
            # (order-invariant comparison)
            pred_elem_set = set()
            for s in range(self.n_slots):
                e = pred_elements[s].item()
                if e != 0:
                    pred_elem_set.add(e)
            gt_elem_set = set()
            for j in range(n_gt):
                gt_elem_set.add(gt_elements[b, j].item())
            if pred_elem_set == gt_elem_set:
                total_set_exact += 1

        # Average over batch
        element_loss = element_loss / B
        fraction_loss = fraction_loss / max(B, 1)
        presence_loss = presence_loss / B

        total = (self.element_loss_weight * element_loss +
                 self.fraction_loss_weight * fraction_loss +
                 self.presence_bce_weight * presence_loss)

        return {
            'total': total,
            'element_loss': element_loss.detach(),
            'fraction_loss': fraction_loss.detach(),
            'presence_loss': presence_loss.detach(),
            'element_accuracy': total_elem_correct / max(total_elem_count, 1),
            'fraction_mse': total_frac_mse / max(total_frac_count, 1),
            'set_exact': total_set_exact / B,
        }
