import torch


def resolve_device(requested_device, logger):
    """Resolve a requested training device while logging accelerator fallback."""
    if requested_device in ("auto", "cuda") and torch.cuda.is_available():
        logger.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    if requested_device == "cuda":
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")
