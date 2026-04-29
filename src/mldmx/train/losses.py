import torch.nn.functional as F


def soft_label_cross_entropy(logits, target):
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
