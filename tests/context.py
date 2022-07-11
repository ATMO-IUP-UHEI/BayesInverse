import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bayesinverse import *

from pathlib import Path
from scipy import sparse
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def test_data_path(tmp_path_factory):
    """
    Creates a temporary data directory with pseudo-data for tests.

    Returns
    -------
    data_path : pathlib Path
        Parent of temporary data directory.
    """

    # Configure tmp data path
    data_path = tmp_path_factory.mktemp("data")
    return data_path
