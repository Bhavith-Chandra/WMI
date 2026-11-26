import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3.model import M3, VQTokenizer  # noqa: E402
from m3.provenance import INPUTS  # noqa: E402

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def tok():
    return VQTokenizer.from_checkpoint(INPUTS["tokenizer"])


@pytest.fixture(scope="session")
def tiny():
    return M3.from_checkpoint(INPUTS["tiny"])


@pytest.fixture(scope="session")
def example():
    codes = torch.from_numpy(np.load(INPUTS["example_prompt_ids"]).astype(np.int64))
    c = np.load(INPUTS["example_conditioning"])
    return codes, torch.tensor(c["lob"]).float(), torch.tensor(c["initial_state_float"]).float()
