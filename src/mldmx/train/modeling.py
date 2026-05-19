import torch

def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def model_kwargs_from_args(args, input_dim):
    return {
        "input_dim": input_dim,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "out_dim": len(args.valid_labels),
    }
