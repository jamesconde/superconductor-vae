"""
Environment-aware configuration for DataLoader and training settings.

Auto-detects the runtime environment (WSL2, Colab, bare Linux) and GPU class,
then returns a flat config dict with tuned DataLoader parameters. This prevents
OOM crashes on WSL2 (7.5GB RAM, 22 CPU threads spawning too many workers) while
allowing Colab A100 and bare Linux machines to use more aggressive settings.

Usage:
    from superconductor.utils.env_config import detect_environment
    env = detect_environment()  # prints one-line banner
    # env['num_workers'], env['pin_memory'], etc.
"""

import os
import platform


def _detect_runtime() -> str:
    """Detect runtime environment.

    Priority:
        1. Colab  — ``import google.colab`` succeeds or COLAB_RELEASE_TAG set
        2. WSL2   — 'microsoft' in /proc/version
        3. Linux  — platform.system() == 'Linux'
        4. Other  — conservative fallback
    """
    # Colab check
    if os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass

    # WSL2 check
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                return "wsl2"
    except (FileNotFoundError, PermissionError):
        pass

    # Bare Linux
    if platform.system() == "Linux":
        return "linux"

    return "other"


def _classify_gpu() -> dict:
    """Return GPU name, VRAM in GB, and a size class.

    Size classes:
        'xlarge' — 70GB+ (A100-80GB, H100-80GB)
        'large'  — 38-70GB (A100-40GB)
        'medium' — 14-38GB (V100, T4, L4, RTX 3090/4090)
        'small'  — <14GB (RTX 4060, GTX 1080 Ti, etc.)
        'none'   — no CUDA GPU detected
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"name": "none", "vram_gb": 0.0, "class": "none"}
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        name = props.name
        if vram_gb >= 70:  # A100-80GB reports ~78GB usable
            gpu_class = "xlarge"
        elif vram_gb >= 38:  # A100-40GB reports 39.6GB usable
            gpu_class = "large"
        elif vram_gb >= 14:
            gpu_class = "medium"
        else:
            gpu_class = "small"
        return {"name": name, "vram_gb": round(vram_gb, 1), "class": gpu_class}
    except Exception:
        return {"name": "unknown", "vram_gb": 0.0, "class": "none"}


def _get_system_ram_gb() -> float:
    """Return total system RAM in GB (best-effort)."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Value is in kB
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0.0


def _get_cpu_count() -> int:
    """Return usable CPU count."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def detect_environment() -> dict:
    """Auto-detect environment and return a flat config dict.

    Returns a dict with keys:
        environment         — 'wsl2', 'colab', 'linux', 'other'
        gpu                 — dict with 'name', 'vram_gb', 'class'
        system_ram_gb       — total system RAM
        num_workers         — DataLoader num_workers
        pin_memory          — DataLoader pin_memory
        persistent_workers  — DataLoader persistent_workers
        prefetch_factor     — DataLoader prefetch_factor (None when workers=0)
        batch_size_multiplier — multiplier vs default batch size
        use_torch_compile   — whether torch.compile is beneficial
        summary             — human-readable one-line summary
    """
    runtime = _detect_runtime()
    gpu = _classify_gpu()
    ram_gb = _get_system_ram_gb()
    cpus = _get_cpu_count()

    # --- Settings by environment ---

    if runtime == "wsl2":
        num_workers = 2
        pin_memory = False
        persistent_workers = False
        prefetch_factor = 1
        batch_size_multiplier = 1.0
        use_torch_compile = True
        compile_mode = None  # Use TRAIN_CONFIG default (reduce-overhead works on local)

    elif runtime == "colab":
        if gpu["class"] == "xlarge":
            # A100-80GB / H100-80GB (70GB+)
            # V12.43 Net2Net expansion (d_model 512→576, dim_ff 2048→2304) increased
            # per-sample memory ~15-20%. Old batch=2100 (50x) OOM'd at 76.7/79.3GB.
            # New: 25x * 2 accum = same effective batch (2100), half peak memory.
            # 42 * 25 = 1050 per step, ~45 steps/epoch (52K / 1050).
            num_workers = min(8, cpus - 1) if cpus > 1 else 0
            pin_memory = True
            persistent_workers = True
            prefetch_factor = 4
            batch_size_multiplier = 25.0  # 42 * 25 = 1050 (halved from 50 post-expansion)
            accumulation_steps = 1        # V16.0: No accumulation — RL-aware scaling handles batch sizing
            n_samples_rloo = 4            # 4 samples: proven effective, save VRAM for batch
            selective_backprop = False     # All samples get full gradients
            use_torch_compile = True
            compile_mode = "reduce-overhead"
        elif gpu["class"] == "large":
            # A100-40GB / H100-40GB (38-70GB)
            # V12.43 Net2Net expansion: batch=504 (12x) was borderline on 40GB.
            # Reduced to 8x with 2-step accumulation: 42*8=336 per step,
            # effective=672. ~157 steps/epoch (52K / 336).
            num_workers = min(8, cpus - 1) if cpus > 1 else 0
            pin_memory = True
            persistent_workers = True
            prefetch_factor = 3
            batch_size_multiplier = 8.0   # 42 * 8 = 336 (reduced from 12 post-expansion)
            accumulation_steps = 1        # V16.0: No accumulation — RL-aware scaling handles batch sizing
            n_samples_rloo = 4            # 4 samples: proven effective
            selective_backprop = False     # All samples get full gradients (no skipping)
            use_torch_compile = True
            compile_mode = "reduce-overhead"
        elif gpu["class"] in ("medium", "small") and gpu["class"] != "none":
            # T4 / V100 / L4 (14-40GB) or small Colab GPU
            num_workers = min(2, cpus - 1) if cpus > 1 else 0
            pin_memory = True
            persistent_workers = True
            prefetch_factor = 2
            batch_size_multiplier = 1.5
            use_torch_compile = True
            compile_mode = "default"
        else:
            # Colab CPU runtime
            num_workers = 0
            pin_memory = False
            persistent_workers = False
            prefetch_factor = None
            batch_size_multiplier = 0.5
            use_torch_compile = False
            compile_mode = None

    elif runtime == "linux":
        if gpu["class"] in ("xlarge", "large", "medium") and ram_gb >= 32:
            # Big bare Linux workstation / server
            num_workers = min(8, cpus - 1) if cpus > 1 else 0
            pin_memory = True
            persistent_workers = True
            prefetch_factor = 2
            batch_size_multiplier = 2.5
            use_torch_compile = True
            compile_mode = None  # Use TRAIN_CONFIG default
        else:
            # Small bare Linux or no GPU
            num_workers = min(2, cpus - 1) if cpus > 1 else 0
            pin_memory = ram_gb >= 16 and gpu["class"] != "none"
            persistent_workers = False
            prefetch_factor = 1
            batch_size_multiplier = 1.0
            use_torch_compile = True
            compile_mode = None

    else:
        # Conservative fallback (macOS, Windows native, etc.)
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
        batch_size_multiplier = 1.0
        use_torch_compile = False
        compile_mode = None

    # prefetch_factor must be None when num_workers == 0
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    summary = (
        f"{runtime.upper()} | GPU: {gpu['name']} ({gpu['vram_gb']}GB) | "
        f"RAM: {ram_gb}GB | workers={num_workers}, pin_memory={pin_memory}"
    )

    # These are only overridden for A100+ (extra VRAM available)
    _accum = locals().get('accumulation_steps', None)
    _n_samples = locals().get('n_samples_rloo', None)
    _sel_bp = locals().get('selective_backprop', None)

    config = {
        "environment": runtime,
        "gpu": gpu,
        "system_ram_gb": ram_gb,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "batch_size_multiplier": batch_size_multiplier,
        "accumulation_steps": _accum,       # None = use TRAIN_CONFIG default, int = override
        "n_samples_rloo": _n_samples,       # None = use TRAIN_CONFIG default, int = override
        "selective_backprop": _sel_bp,       # None = use TRAIN_CONFIG default, bool = override
        "use_torch_compile": use_torch_compile,
        "compile_mode": compile_mode,  # V12.20: None = use TRAIN_CONFIG, str = override
        "summary": summary,
    }

    print(f"[env] {summary}")
    return config
