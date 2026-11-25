import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed python, numpy and torch; force deterministic torch kernels.

    Returns a numpy Generator that callers should thread through explicitly
    rather than relying on global state.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:  # pragma: no cover
        pass
    return np.random.default_rng(seed)
