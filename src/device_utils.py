"""
Device helpers so the same code runs on NVIDIA GPUs (cuda), Apple Silicon (mps) and the CPU.

Why this exists: GPU work is asynchronous. A timing taken without waiting for the device to finish measures only how fast the work was
QUEUED. torch.cuda.synchronize() does that wait on NVIDIA; Apple's equivalent is torch.mps.synchronize(). Code that only synchronised
on "cuda" would under-report times on a Mac.
"""
import torch


def sync_device(device) -> None:
    """Wait until all queued work on `device` has finished (no-op on the CPU)."""
    d = str(device)
    try:
        if d.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif d.startswith("mps") and hasattr(torch, "mps"):
            torch.mps.synchronize()
    except Exception:
        pass


def empty_device_cache(device) -> None:
    """Hand cached, unused device memory back to the system (no-op on the CPU)."""
    d = str(device)
    try:
        if d.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif d.startswith("mps") and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


def device_memory_gb(device):
    """(free_gb, total_gb) for a CUDA device, read live from the driver; None for mps and cpu.
    Apple Silicon has unified memory shared with the rest of the system, so there is no separate 'GPU memory' to read."""
    try:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1e9, total / 1e9
    except Exception:
        pass
    return None


def describe_device(device) -> str:
    """One readable line saying what the code will run on."""
    d = str(device)
    try:
        if d.startswith("cuda") and torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    if d.startswith("mps"):
        return "mps (Apple Silicon GPU, unified memory)"
    return "cpu"
