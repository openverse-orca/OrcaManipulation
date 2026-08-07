"""HDF5 dataset chunk tuples: HDF5 requires every chunk dimension to be positive."""

from __future__ import annotations

from typing import Optional, Tuple, Union


def h5py_chunks_if_valid(
    chunks: Union[Tuple[int, ...], Tuple[()]],
) -> Optional[Tuple[int, ...]]:
    """
    Return ``chunks`` for h5py ``create_dataset``, or ``None`` if any chunk
    dimension is non-positive (illegal in HDF5; use default/auto chunking).
    """
    if not chunks:
        return None
    t = tuple(int(x) for x in chunks)
    if any(x <= 0 for x in t):
        return None
    return t
