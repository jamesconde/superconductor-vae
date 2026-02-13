"""
V12.29: Training Manifest System

Embeds version, config, dataset, and architecture fingerprints into checkpoints
and latent caches. Detects config drift when loading checkpoints saved by
different code/config versions.

Usage:
    from superconductor.utils.manifest import build_manifest, check_config_drift

    manifest = build_manifest(
        model_config=MODEL_CONFIG,
        train_config=TRAIN_CONFIG,
        dataset_fingerprint={'n_rows': 46645, 'n_cols': 150, 'magpie_dim': 145},
        encoder=encoder,
        package_version='0.2.0',
    )

    # Embed in checkpoint
    checkpoint['manifest'] = manifest

    # On load, check for drift
    warnings = check_config_drift(checkpoint['manifest'], current_manifest)
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def get_git_info() -> dict:
    """Returns {'commit': str, 'branch': str, 'dirty': bool} or {} if not a git repo."""
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        branch = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True, text=True, timeout=5
        )
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if commit.returncode != 0:
            return {}
        return {
            'commit': commit.stdout.strip(),
            'branch': branch.stdout.strip(),
            'dirty': len(status.stdout.strip()) > 0,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def get_environment_info() -> dict:
    """Returns {'python': str, 'torch': str, 'cuda': str, 'gpu_name': str}."""
    info = {
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda': torch.version.cuda or 'none',
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
    else:
        info['gpu_name'] = 'none'
    return info


def get_model_architecture_fingerprint(model: nn.Module) -> dict:
    """Returns {param_name: list(shape)} for all parameters -- a structural fingerprint."""
    return {
        name: list(param.shape)
        for name, param in model.named_parameters()
    }


def compute_config_hash(config: dict) -> str:
    """Stable SHA-256 hash of a config dict (sorted keys, JSON-serialized).

    Non-serializable values (e.g., Path objects) are converted to strings.
    """
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [_make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        else:
            return str(obj)

    serializable = _make_serializable(config)
    config_str = json.dumps(serializable, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()


def build_manifest(
    model_config: dict,
    train_config: dict,
    dataset_fingerprint: dict,
    encoder: nn.Module,
    package_version: str,
) -> dict:
    """Assembles the complete manifest dict for embedding in checkpoints.

    Args:
        model_config: MODEL_CONFIG dict (stored in full -- small, affects architecture).
        train_config: TRAIN_CONFIG dict (only hash stored -- large, changes between runs).
        dataset_fingerprint: Dict with keys like n_rows, n_cols, magpie_dim, csv_path.
        encoder: The encoder model (for architecture fingerprint).
        package_version: From superconductor.__version__.

    Returns:
        Manifest dict suitable for embedding in checkpoint_data or cache_data.
    """
    return {
        'version': package_version,
        'git': get_git_info(),
        'environment': get_environment_info(),
        'model_config': dict(model_config),
        'model_config_hash': compute_config_hash(model_config),
        'train_config_hash': compute_config_hash(train_config),
        'dataset': dict(dataset_fingerprint),
        'architecture': get_model_architecture_fingerprint(encoder),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def check_config_drift(saved_manifest: dict, current_manifest: dict) -> List[str]:
    """Compare saved vs current manifest. Returns list of warning strings.

    Tiered warnings:
    - [CRITICAL]: architecture fingerprint mismatch (param shapes differ)
    - [WARNING]: model_config_hash or train_config_hash changed
    - [INFO]: git commit changed, environment changed, dataset changed
    """
    warnings = []

    # CRITICAL: Architecture fingerprint mismatch
    saved_arch = saved_manifest.get('architecture', {})
    current_arch = current_manifest.get('architecture', {})
    if saved_arch and current_arch:
        # Check for missing/added parameters
        saved_keys = set(saved_arch.keys())
        current_keys = set(current_arch.keys())
        added = current_keys - saved_keys
        removed = saved_keys - current_keys
        if added:
            warnings.append(f"[CRITICAL] New parameters added: {sorted(added)}")
        if removed:
            warnings.append(f"[CRITICAL] Parameters removed: {sorted(removed)}")

        # Check for shape mismatches on shared parameters
        for key in saved_keys & current_keys:
            if saved_arch[key] != current_arch[key]:
                warnings.append(
                    f"[CRITICAL] Shape mismatch for '{key}': "
                    f"checkpoint {saved_arch[key]} vs current {current_arch[key]}"
                )

    # WARNING: Config hash changes
    if (saved_manifest.get('model_config_hash') and
            current_manifest.get('model_config_hash') and
            saved_manifest['model_config_hash'] != current_manifest['model_config_hash']):
        # Show what changed in model_config
        diff_keys = []
        saved_mc = saved_manifest.get('model_config', {})
        current_mc = current_manifest.get('model_config', {})
        for key in set(list(saved_mc.keys()) + list(current_mc.keys())):
            if saved_mc.get(key) != current_mc.get(key):
                diff_keys.append(f"{key}: {saved_mc.get(key)} -> {current_mc.get(key)}")
        warnings.append(
            f"[WARNING] model_config changed: {'; '.join(diff_keys) if diff_keys else 'hash differs'}"
        )

    if (saved_manifest.get('train_config_hash') and
            current_manifest.get('train_config_hash') and
            saved_manifest['train_config_hash'] != current_manifest['train_config_hash']):
        warnings.append("[WARNING] train_config changed since checkpoint was saved")

    # INFO: Git changes
    saved_git = saved_manifest.get('git', {})
    current_git = current_manifest.get('git', {})
    if saved_git and current_git:
        if saved_git.get('commit') != current_git.get('commit'):
            warnings.append(
                f"[INFO] Git commit changed: "
                f"{saved_git.get('commit', '?')[:8]} -> {current_git.get('commit', '?')[:8]}"
            )

    # INFO: Environment changes
    saved_env = saved_manifest.get('environment', {})
    current_env = current_manifest.get('environment', {})
    if saved_env and current_env:
        env_diffs = []
        for key in ['python', 'torch', 'cuda']:
            if saved_env.get(key) != current_env.get(key):
                env_diffs.append(f"{key}: {saved_env.get(key)} -> {current_env.get(key)}")
        if env_diffs:
            warnings.append(f"[INFO] Environment changed: {'; '.join(env_diffs)}")

    # INFO: Dataset changes
    saved_ds = saved_manifest.get('dataset', {})
    current_ds = current_manifest.get('dataset', {})
    if saved_ds and current_ds:
        ds_diffs = []
        for key in ['n_rows', 'n_cols', 'magpie_dim']:
            if saved_ds.get(key) != current_ds.get(key):
                ds_diffs.append(f"{key}: {saved_ds.get(key)} -> {current_ds.get(key)}")
        if ds_diffs:
            warnings.append(f"[INFO] Dataset changed: {'; '.join(ds_diffs)}")

    return warnings
