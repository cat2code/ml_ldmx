from collections import Counter


def count_classes(events, indices):
    counter = Counter()
    ordered_indices = (
        events.order_indices_for_access(indices)
        if hasattr(events, "order_indices_for_access")
        else indices
    )
    for idx in ordered_indices:
        counter.update(events[idx]["physical_y"].tolist())
    return dict(sorted(counter.items()))


def target_order_counts(events, indices):
    counter = Counter()
    ordered_indices = (
        events.order_indices_for_access(indices)
        if hasattr(events, "order_indices_for_access")
        else indices
    )
    for idx in ordered_indices:
        order = events[idx].get("target_label_order")
        if order is not None:
            counter.update([tuple(order)])
    return {str(key): value for key, value in sorted(counter.items())}
