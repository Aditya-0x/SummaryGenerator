import pytest
import numpy as np
from mlplo.eval import maybe_limit_split

def test_maybe_limit_split():
    class DummyDataset:
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
        def select(self, indices):
            return list(indices)
            
    ds = DummyDataset(10)
    assert maybe_limit_split(ds, None) == ds
    assert maybe_limit_split(ds, 15) == ds
    assert maybe_limit_split(ds, 5) == list(range(5))
