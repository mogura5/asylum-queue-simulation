"""Make modules in src/ importable when pytest runs from the repository root."""

import sys
from pathlib import Path
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

TEST_PARAMS = {
    "arrival_rate_lambda": 50.0,
    "service_time_mean": 20.0,
    "service_time_std": 10.0,
    "grant_rate": 0.10,
    "appeal_rate": 0.30,
    "p_rep": 0.6,
    "p_detained": 0.1,
    "p_renege": 0.30,
    "p_criminal": 0.05,
}
 
 
@pytest.fixture
def test_params():
    return dict(TEST_PARAMS)