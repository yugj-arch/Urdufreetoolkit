import os
import sys

import pytest

# Make the app package importable no matter where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def pytest_collection_modifyitems(config, items):
    """Skip providers marked ``@pytest.mark.heavy`` unless RUN_HEAVY=1 — keeps the
    default suite fast and free of multi-GB model downloads."""
    if os.environ.get("RUN_HEAVY"):
        return
    skip_heavy = pytest.mark.skip(reason="heavy provider; set RUN_HEAVY=1 to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
