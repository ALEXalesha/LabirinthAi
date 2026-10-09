import os

import numpy as np

from aikit import trainlog
from aikit.i18n import Translator


def test_log_lines_go_to_the_file_at_once(tmp_path):
    log = trainlog.TrainLog(tmp_path / "logs" / "train.log", echo=False)
    log("партий 100")
    log("партий 200")
    lines = (tmp_path / "logs" / "train.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[1].endswith("партий 200")


def test_snapshot_round_trip_and_broken_snapshot(tmp_path):
    path = tmp_path / "snap.npz"
    assert trainlog.load_snapshot(path) is None
    trainlog.save_snapshot(path, {"w": np.arange(5.0)}, {"games": 12, "имя": "сеть"})
    arrays, meta = trainlog.load_snapshot(path)
    np.testing.assert_array_equal(arrays["w"], np.arange(5.0))
    assert meta == {"games": 12, "имя": "сеть"}
    path.write_bytes(b"\x00 not a zip")
    assert trainlog.load_snapshot(path) is None


def test_limit_threads_sets_blas_variables(monkeypatch):
    for v in trainlog.THREAD_VARS:
        monkeypatch.delenv(v, raising=False)
    trainlog.limit_threads(3)
    assert all(os.environ[v] == "3" for v in trainlog.THREAD_VARS)
    trainlog.limit_threads(5)                  # без force уже заданное не трогается
    assert os.environ["OMP_NUM_THREADS"] == "3"
    trainlog.limit_threads(5, force=True)
    assert os.environ["OMP_NUM_THREADS"] == "5"


def test_translate_with_fallbacks():
    t = Translator({"ru": {"a": "привет, {name}", "b": "только ру"}, "en": {"a": "hello, {name}"}}, "en")
    assert t("a", name="мир") == "hello, мир"
    assert t("b") == "только ру"
    assert t("нет") == "нет"
    assert t.missing() == {"en": ["b"]}
