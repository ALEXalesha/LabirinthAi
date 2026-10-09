import json

import pytest

from aikit import settings as st
from aikit.stats import Stats

SCHEMA = {"theme": ("dark", st.choice("dark", "light")), "volume": (70, st.int_range(0, 100)),
          "on": (False, st.boolean())}


def test_defaults_without_file(tmp_path):
    s = st.Settings(SCHEMA, tmp_path / "s.json")
    assert (s["theme"], s["volume"], s["on"]) == ("dark", 70, False)


def test_each_bad_field_falls_back_alone(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"theme": "light", "volume": 500, "on": "да", "лишнее": 1}), encoding="utf-8")
    s = st.Settings(SCHEMA, path)
    assert (s["theme"], s["volume"], s["on"]) == ("light", 70, False)


@pytest.mark.parametrize("junk", ["", "{обрыв", "[1, 2]", "null", "\x00\x01"])
def test_broken_file_gives_defaults(tmp_path, junk):
    path = tmp_path / "s.json"
    path.write_text(junk, encoding="utf-8")
    assert st.Settings(SCHEMA, path)["volume"] == 70


def test_set_saves_at_once_and_refuses_bad_values(tmp_path):
    path = tmp_path / "s.json"
    s = st.Settings(SCHEMA, path)
    s.set("volume", 15)
    assert st.Settings(SCHEMA, path)["volume"] == 15
    with pytest.raises(ValueError):
        s.set("volume", True)          # bool - не число
    with pytest.raises(ValueError):
        s.set("theme", ["dark"])       # нехэшируемое
    assert st.Settings(SCHEMA, path)["volume"] == 15


def test_stats_record_and_streaks(tmp_path):
    path = tmp_path / "stats.json"
    s = Stats(path)
    for r in ("win", "win", "draw", "win"):
        s.record("checkers/strong", r)
    e = Stats(path).entry("checkers/strong")
    assert (e["games"], e["wins"], e["draws"], e["streak"], e["best_streak"]) == (4, 3, 1, 1, 2)
    with pytest.raises(ValueError):
        s.record("checkers/strong", "победа")


def test_stats_totals_by_prefix(tmp_path):
    s = Stats(tmp_path / "x.json")
    s.record("chess/novice", "loss")
    s.record("checkers/novice", "win")
    s.record("checkers/strong", "draw")
    assert s.totals("checkers/")["games"] == 2
    assert s.totals()["games"] == 3


def test_stats_skip_broken_entries(tmp_path):
    path = tmp_path / "stats.json"
    good = {"games": 1, "wins": 1, "losses": 0, "draws": 0, "streak": 1, "best_streak": 1}
    path.write_text(json.dumps({"a": good, "b": {**good, "wins": -1}, "c": 5}), encoding="utf-8")
    assert set(Stats(path).data) == {"a"}
