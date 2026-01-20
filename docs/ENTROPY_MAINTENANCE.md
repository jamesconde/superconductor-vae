# Entropy Maintenance Module

**Location**: `src/superconductor/training/entropy_maintenance.py`

## Overview

Prevents entropy collapse during REINFORCE training by dynamically adjusting exploration incentives. Implements 7 strategies based on recent research (AER, EPO).

## Quick Start

```python
from superconductor.training import EntropyManager, create_entropy_manager

# Create manager with causal strategy (recommended)
entropy_manager = create_entropy_manager(
    strategy='causal',
    target_entropy=0.5,
    min_entropy=0.1,
)

# In training loop
for epoch in range(n_epochs):
    # Get current settings
    entropy_weight = entropy_manager.get_entropy_weight(epoch, current_entropy)
    temperature = entropy_manager.get_temperature(epoch, current_entropy)

    # Use in REINFORCE
    reward = base_reward + entropy_weight * entropy

    # Update with metrics after each epoch
    entropy_manager.update(
        epoch=epoch,
        entropy=current_entropy,
        reward=mean_reward,
        exact_match=exact_match,
    )
```

## Available Strategies

### 1. Causal (`'causal'`) - RECOMMENDED
Diagnoses plateau causes before intervening. Unlike naive "plateau → boost" approaches:
- Checks if entropy actually dropped BEFORE the plateau
- Checks if entropy is currently below minimum threshold
- Uses tiered response based on evidence strength
- Tracks intervention success and adjusts confidence over time

**Evidence Scoring:**
- STRONG (both conditions): entropy dropped AND is low → 2x boost
- WEAK (one condition): entropy dropped OR is low → 1.3x boost
- NONE (neither): plateau but entropy healthy → no boost

**Historical Learning:**
- Tracks if boosts actually break plateaus
- Reduces boost confidence after failed interventions (→ 1.1x minimal boost)

```python
em = create_entropy_manager('causal', target_entropy=0.5, min_entropy=0.1)

# Logs diagnostic info during training:
#   [Entropy] Plateau + strong evidence → 2.0x boost (dropped=True, low=True)
#   [Entropy] Plateau detected but entropy not implicated - no boost
```

### 2. Adaptive (`'adaptive'`)
Dynamically adjusts entropy weight based on:
- Current entropy vs target
- Plateau detection (no improvement)
- Automatically boosts when entropy too low

Note: This strategy boosts on ANY plateau. Consider 'causal' for smarter diagnosis.

```python
em = create_entropy_manager('adaptive', target_entropy=0.5, min_entropy=0.1)
```

### 3. Cyclical (`'cyclical'`)
Temperature warm restarts - periodically boosts temperature to encourage exploration.

```python
em = create_entropy_manager(
    'cyclical',
    temperature_restart_period=50,  # Restart every 50 epochs
    temperature_restart_boost=0.3,  # Boost amount
)
```

### 4. Position Weighted (`'position_weighted'`)
Tracks error rates per sequence position and weights entropy accordingly. High error positions get more exploration.

```python
em = create_entropy_manager('position_weighted', error_position_boost=2.0)

# Update with position error data
em.update(
    epoch=epoch,
    position_errors=error_array,  # (batch, seq_len) binary
    position_mask=mask_array,      # (batch, seq_len) valid positions
)

# Get position weights
weights = em.get_position_weights()  # (max_len,)
```

### 5. Novelty Bonus (`'novelty_bonus'`)
Rewards diverse generations by comparing to history buffer.

```python
em = create_entropy_manager(
    'novelty_bonus',
    novelty_weight=0.1,
    novelty_buffer_size=1000,
)

# Compute novelty bonus for batch
novelty = em.compute_novelty_bonus(generated_tokens)  # List[List[int]]
reward = base_reward + novelty
```

### 6. Uncertainty Guided (`'uncertainty'`)
Uses reward variance to guide exploration. High variance = uncertain = explore more.

```python
em = create_entropy_manager('uncertainty', variance_threshold=0.1)

# Update with batch rewards (for variance estimation)
em.update(rewards_batch=batch_rewards_array)
```

### 7. Composite (`'composite'`)
Combines adaptive + cyclical + uncertainty. Best for complex training.

```python
em = create_entropy_manager('composite')
```

## Integration with train_v12_clean.py

To enable entropy maintenance in training:

```python
# In TRAIN_CONFIG
'entropy_strategy': 'causal',   # Recommended: diagnoses plateau cause before boosting
'target_entropy': 0.5,
'min_entropy': 0.1,

# In training initialization
from superconductor.training import create_entropy_manager
entropy_manager = create_entropy_manager(
    strategy=config['entropy_strategy'],
    target_entropy=config.get('target_entropy', 0.5),
    min_entropy=config.get('min_entropy', 0.1),
)

# In training loop (after RL computation)
if config['rl_weight'] > 0:
    entropy_weight = entropy_manager.get_entropy_weight(epoch, current_entropy)
    # Modify REINFORCE: reward = reward + entropy_weight * entropy

    entropy_manager.update(
        epoch=epoch,
        entropy=current_entropy,
        reward=mean_reward,
        exact_match=exact_match_rate,
    )

# In checkpoint saving
checkpoint['entropy_manager_state'] = entropy_manager.get_state()

# In checkpoint loading
if 'entropy_manager_state' in checkpoint:
    entropy_manager.load_state(checkpoint['entropy_manager_state'])
```

## Key Parameters

### Common Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_entropy` | 0.5 | Entropy level to maintain (nats) |
| `min_entropy` | 0.1 | Critical threshold - strong boost below this |
| `entropy_weight_base` | 0.2 | Base entropy bonus weight |
| `entropy_weight_max` | 1.0 | Maximum entropy weight |
| `plateau_window` | 10 | Epochs to detect training plateau |
| `plateau_threshold` | 0.01 | Improvement threshold (1% relative by default) |
| `plateau_relative` | True | If True, threshold scales with performance |
| `temperature_restart_period` | 50 | Epochs between temp restarts |

### Causal Strategy Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `causal_diagnosis_window` | 10 | Epochs to check entropy trend before plateau |
| `causal_followup_window` | 10 | Epochs to check if boost helped |
| `causal_entropy_drop_threshold` | 0.1 | 10% drop = entropy dropped |
| `causal_min_success_rate` | 0.3 | Min success rate to trust boosts |
| `causal_strong_boost` | 2.0 | Multiplier for strong evidence |
| `causal_weak_boost` | 1.3 | Multiplier for weak evidence |
| `causal_minimal_boost` | 1.1 | Multiplier when history shows poor success |

## Plateau Detection

With `plateau_relative=True` (default), the threshold scales with current performance:

| Performance | Threshold (1%) | Required Improvement |
|-------------|----------------|---------------------|
| 80% exact | 0.8% relative | 0.64% absolute |
| 50% exact | 0.5% relative | 0.25% absolute |
| 20% exact | 0.2% relative | 0.04% absolute |

This means:
- At high performance, larger absolute improvements are required
- At low performance, small gains are considered progress
- Reduces hyperparameter tuning (threshold adapts automatically)

## Checkpointing

```python
# Save
state = entropy_manager.get_state()

# Load
entropy_manager.load_state(state)
```

## Monitoring

```python
# Get detailed info
info = entropy_manager.get_info(epoch, current_entropy)
print(f"Strategy: {info['strategy']}")
print(f"Current weight: {info['entropy_weight']:.4f}")
print(f"Plateau detected: {info.get('plateau_detected', False)}")
```
