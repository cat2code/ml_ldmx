import argparse
from pathlib import Path
import torch
import torch.nn as nn

from mldmx.io.root_reader import RootSource, read_branches
from mldmx.io.branches import BRANCHES
from mldmx.data.tensorize import ecal_hits_to_padded_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=10)
    parser.add_argument("--max-hits", type=int, default=256)
    args = parser.parse_args()

    root_path = Path(args.root_file).resolve()
    print(f"Opening ROOT file: {root_path}")

    if not root_path.exists():
        raise FileNotFoundError(f"ROOT file not found: {root_path}")

    source = RootSource(path=str(root_path), tree_name="LDMX_Events")
    ecal = BRANCHES["ecal"]["simhits_pileup"]
    branch_names = list(ecal.values())

    print("Reading branches:")
    for b in branch_names:
        print("  ", b)

    arrays = read_branches(
        source,
        branch_names=branch_names,
        entry_start=args.start,
        entry_stop=args.stop,
    )

    X, mask = ecal_hits_to_padded_tensor(arrays, ecal, max_hits=args.max_hits)

    print("X shape:", X.shape)
    print("mask shape:", mask.shape)
    print("valid hits per event:", mask.sum(dim=1).tolist())

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )

    out = model(X)
    print("model output shape:", out.shape)


if __name__ == "__main__":
    main()