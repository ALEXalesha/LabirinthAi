import os
import sys

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Без экрана: Qt рисует в память. Ставится до любого импорта Qt.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
if os.path.isdir(_fonts):
    os.environ.setdefault("QT_QPA_FONTDIR", _fonts)

from hypothesis import settings  # noqa: E402

settings.register_profile("default", max_examples=400, deadline=None)
settings.register_profile("thorough", max_examples=4000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
