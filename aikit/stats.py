"""Статистика игрока: по ключу (игра/уровень) - партии, победы, поражения, ничьи, серия побед.
Запись сразу; битый файл - пустая статистика, битая запись пропускается."""
from aikit.storage import is_int, load_json, save_json

RESULTS = ("win", "loss", "draw")
FIELDS = ("games", "wins", "losses", "draws", "streak", "best_streak")


def empty():
    return {f: 0 for f in FIELDS}


class Stats:
    def __init__(self, path=None, data=None):
        self.path = path
        self.data = data if data is not None else {}
        if path is not None and data is None:
            raw = load_json(path)
            if isinstance(raw, dict):
                for k, e in raw.items():
                    if isinstance(e, dict) and all(is_int(e.get(f)) and e[f] >= 0 for f in FIELDS):
                        self.data[k] = {f: e[f] for f in FIELDS}

    def entry(self, key):
        return dict(self.data.get(key, empty()))

    def record(self, key, result):
        """result - с точки зрения игрока: 'win', 'loss' или 'draw'. Ничья и поражение
        обрывают серию побед."""
        if result not in RESULTS:
            raise ValueError(f"неизвестный итог партии: {result!r}")
        e = self.entry(key)
        e["games"] += 1
        e[{"win": "wins", "loss": "losses", "draw": "draws"}[result]] += 1
        e["streak"] = e["streak"] + 1 if result == "win" else 0
        e["best_streak"] = max(e["best_streak"], e["streak"])
        self.data[key] = e
        self.save()
        return e

    def totals(self, prefix=""):
        out = empty()
        chosen = [e for k, e in self.data.items() if k.startswith(prefix)]
        for e in chosen:
            for f in ("games", "wins", "losses", "draws"):
                out[f] += e[f]
        out["best_streak"] = max([e["best_streak"] for e in chosen] or [0])
        return out

    def save(self):
        if self.path is None:
            return False
        return save_json(self.path, self.data)
