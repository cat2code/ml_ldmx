import torch


def resolve_device(requested_device, logger):
    use_cuda = requested_device in ("auto", "cuda") and torch.cuda.is_available()
    if requested_device == "cuda" and not use_cuda:
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
    if use_cuda:
        logger.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
    return torch.device("cuda" if use_cuda else "cpu")
