import torch


def build_knn_graph(pos: torch.Tensor, k: int = 8) -> torch.Tensor:
    num_nodes = pos.size(0)

    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)

    dist = torch.cdist(pos, pos)  # [N, N]
    knn = dist.topk(k=min(k + 1, num_nodes), largest=False).indices  # includes self

    edges = []
    for i in range(num_nodes):
        for j in knn[i]:
            j = j.item()
            if i != j:
                edges.append([i, j])

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_ecal_tpad_context_graph(
    ecal_pos: torch.Tensor,
    tpad_tensor: torch.Tensor,
    k_ecal: int = 8,
    k_tpad_to_ecal: int = 16,
) -> torch.Tensor:
    """
    Build ECal kNN edges plus bidirectional TriggerPadTracks-to-ECal context edges.

    ECal nodes are expected first in the combined node tensor, followed by
    TriggerPadTracks nodes. TriggerPadTracks centroid_ is a 1D y-like coordinate,
    so context edges are chosen by absolute distance to ECal y, conventionally
    ecal_pos[:, 1].
    """

    num_ecal = ecal_pos.size(0)
    num_tpad = tpad_tensor.size(0)

    ecal_edges = build_knn_graph(ecal_pos, k=k_ecal)
    if num_ecal == 0 or num_tpad == 0:
        return ecal_edges

    k_context = min(k_tpad_to_ecal, num_ecal)
    context_edges = []
    ecal_y = ecal_pos[:, 1]
    tpad_centroid = tpad_tensor[:, 0]

    for itpad in range(num_tpad):
        tpad_node = num_ecal + itpad
        distances = torch.abs(ecal_y - tpad_centroid[itpad])
        nearest_ecal = distances.topk(k=k_context, largest=False).indices
        for iecal in nearest_ecal.tolist():
            context_edges.append([tpad_node, iecal])
            context_edges.append([iecal, tpad_node])

    if not context_edges:
        return ecal_edges

    context_edges = torch.tensor(context_edges, dtype=torch.long).t().contiguous()
    if ecal_edges.numel() == 0:
        return context_edges
    return torch.cat([ecal_edges, context_edges], dim=1)
