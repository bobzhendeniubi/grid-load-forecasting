import pandas as pd
import pytest

from src.data import load_demand
from src.features import build_features


@pytest.fixture(scope="session")
def raw():
    import os
    os.environ["HISTORY_DAYS"] = "90"
    return load_demand(use_cache=False)


@pytest.fixture(scope="session")
def feat(raw):
    return build_features(raw)
