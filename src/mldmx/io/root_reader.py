# mldmx/src/mldmx/io/root_reader.py

from dataclasses import dataclass
import uproot

'''
This module provides utilities for reading ROOT files using uproot with awkward arrays. 
It defines a RootSource dataclass to specify the file and tree, and a read_branches function to read specified branches into awkward arrays.
'''

@dataclass(frozen=True)
class RootSource:
    path: str
    tree_name: str = "LDMX_Events"


def read_branches(source: RootSource, branch_names, entry_start=0, entry_stop=None):
    with uproot.open(source.path) as f:
        tree = f[source.tree_name]
        arrays = tree.arrays(branch_names, entry_start=entry_start, entry_stop=entry_stop, library="ak")
    return arrays