
def chunks(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]
