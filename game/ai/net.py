"""Сеть на numpy: вход features.encode -> два скрытых слоя -> 4 хода и оценка.

Голова ходов - логиты N E S W в развёрнутом окне; запрещённые ходы (стена у своей клетки)
получают -inf до softmax. Голова оценки - сколько примерно ещё идти (в единицах награды
обучения), для «Мыслей ИИ» и обучения.
"""
import json
import os

import numpy as np

from aikit.nn import io as nnio
from aikit.nn.layers import Dense, ReLU, Sequential
from aikit.nn.losses import softmax
from game.ai import features as F

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
STRONG = "labirinth-policy.npz"
EARLY = "labirinth-policy-early.npz"         # чистый PPO в начале обучения («Смотреть»)
PPO = "labirinth-policy-ppo.npz"             # чистый PPO, обучен до конца (средний уровень)
PASSPORT = "labirinth-policy.json"
NEG = -1e9


class PolicyNet:
    def __init__(self, hidden=(128, 128), seed=0, nin=F.SIZE):
        rng = np.random.default_rng(seed)
        self.hidden = tuple(hidden)
        self.nin = nin
        self.hints = nin == F.SIZE_HINT          # сеть берёт и подсказку-прикидку пути
        items = []
        n = nin
        for h in self.hidden:
            items += [Dense(n, h, rng), ReLU()]
            n = h
        self.body = Sequential(items)
        self.pi = Dense(n, 4, rng)
        self.v = Dense(n, 1, rng)
        self.pi.W *= 0.01                 # почти равные ходы в начале
        self.v.W *= 0.01

    def parts(self):
        return {"body": self.body, "pi": self.pi, "v": self.v}

    def params(self):
        return [(f"{k}.{n}", p, g) for k, part in self.parts().items() for n, p, g in part.params()]

    def count(self):
        return int(sum(p.size for _, p, _ in self.params()))

    def forward(self, x, legal):
        """(логиты с -inf у запрещённых (B, 4), оценка (B,))."""
        h = self.body.forward(x.astype(np.float32, copy=False))
        logits = self.pi.forward(h)
        value = self.v.forward(h)[:, 0]
        self.h = h
        return np.where(legal, logits, NEG).astype(np.float32), value

    def backward(self, dlogits, dvalue):
        dh = self.pi.backward(dlogits) + self.v.backward(dvalue[:, None].astype(np.float32))
        self.body.backward(dh)

    def probs(self, x, legal):
        logits, value = self.forward(x, legal)
        return softmax(logits), value

    def save(self, path, meta=None):
        arrays = nnio.collect(self.parts())
        arrays["__hidden__"] = np.array(self.hidden)
        arrays["__nin__"] = np.array(self.nin)
        nnio.save(path, arrays)
        if meta is not None:
            with open(os.path.splitext(path)[0] + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        arrays = nnio.load(path)
        nin = int(arrays["__nin__"]) if "__nin__" in arrays else F.SIZE
        net = cls(tuple(int(h) for h in arrays["__hidden__"]), nin=nin)
        nnio.restore(net.parts(), arrays)
        return net

    def copy_from(self, other):
        for (_, a, _), (_, b, _) in zip(self.params(), other.params()):
            a[...] = b


_CACHE = {}


def load_model(name):
    """Сеть из game/models по имени файла; None, если её ещё нет (не обучена)."""
    path = os.path.join(MODEL_DIR, name)
    if name not in _CACHE:
        _CACHE[name] = PolicyNet.load(path) if os.path.exists(path) else None
    return _CACHE[name]


def load_passport():
    path = os.path.join(MODEL_DIR, PASSPORT)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
