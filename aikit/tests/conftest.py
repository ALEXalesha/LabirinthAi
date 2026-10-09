import os
import sys

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402

from aikit import offscreen  # noqa: E402

offscreen.setup()          # без экрана: Qt рисует в память, до любого импорта Qt


@pytest.fixture
def qapp():
    """Одно QApplication на весь прогон: второе Qt создать не даст."""
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
