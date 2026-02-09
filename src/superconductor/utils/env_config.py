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
        'large'  — 40GB+ (A100, H100, etc.)
        'medium' — 14-40GB (V100, T4, L4, RTX 3090/4090)
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
        if vram_gb >= 40:
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

    elif runtime == "colab":
        if gpu["class"] == "large":
            # A100 / H100 (40GB+)
            num_workers = min(4, cpus - 1) if cpus > 1 else 0
            pin_memory = True
            persistent_workers = True
            prefetch_factor = 2
            batch_size_multiplier = 2.5
            use_torch_compile = False
        elif gpu["class"] in ("medium", "small") and gpu["class"] != "none":
            # T4 / V100 / L4 (14-40GB) or small Colab GPU
            num_workers = min(2, cpus - 1) if cpus > 1 else 0
            pin_memory = True
            persistent_workers = True
            prefetch_factor = 2
            batch_size_multiplier = 1.5
            use_torch_compile = False
        else:
            # Colab CPU runtime
            num_workers = 0
            pin_memory = False
            persistent_workers = False
            prefetch_factor = None
            batch_size_multiplier = 0.5
            use_torch_compile = False

    elif runtime == "linux":
        if gpu["class"] in ("large", "medium") and ram_gb >= 32:
            # Big bare Linux workstation / server
            num_workers = min(8, cpus - 1) if cpus > 1 else 0
            pin_memory = True
            persistent_workers = True
            prefetch_factor = 2
            batch_size_multiplier = 2.5
            use_torch_compile = True
        else:
            # Small bare Linux or no GPU
            num_workers = min(2, cpus - 1) if cpus > 1 else 0
            pin_memory = ram_gb >= 16 and gpu["class"] != "none"
            persistent_workers = False
            prefetch_factor = 1
            batch_size_multiplier = 1.0
            use_torch_compile = True

    else:
        # Conservative fallback (macOS, Windows native, etc.)
        num_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
        batch_size_multiplier = 1.0
        use_torch_compile = False

    # prefetch_factor must be None when num_workers == 0
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    summary = (
        f"{runtime.upper()} | GPU: {gpu['name']} ({gpu['vram_gb']}GB) | "
        f"RAM: {ram_gb}GB | workers={num_workers}, pin_memory={pin_memory}"
    )

    config = {
        "environment": runtime,
        "gpu": gpu,
        "system_ram_gb": ram_gb,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "batch_size_multiplier": batch_size_multiplier,
        "use_torch_compile": use_torch_compile,
        "summary": summary,
    }

    print(f"[env] {summary}")
    return config
