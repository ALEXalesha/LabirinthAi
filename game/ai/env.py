"""Пачка лабиринтов для обучения и проверок: B проходов идут разом, массивами numpy.

Устроена как fog.Run, только сразу на B лабиринтов: настоящие проходы (true) спрятаны
внутри, наружу (в features.encode) идёт только память - known, mem, visits, pos.
Закон в тестах: память пачки совпадает с памятью fog.Run на тех же лабиринтах и ходах.

Лабиринты разных размеров лежат в общих массивах (Hm, Wm); за размером поля - пусто.
"""
import numpy as np

from game.ai import features as F
from game.core.fog import VIEW
from game.core.generators import generate

TRAIN_KINDS = ("dfs", "prim", "kruskal", "wilson")
STEP_COST = 0.01
SHAPING = 0.0
NEW_CELL = 0.005


class BatchEnv:
    def __init__(self, n, hmax, wmax, shaping=SHAPING, new_cell=NEW_CELL):
        self.n, self.Hm, self.Wm = n, hmax, wmax
        self.shaping, self.new_cell = shaping, new_cell
        self.true = np.zeros((n, hmax, wmax), np.uint8)
        # память с полями шириной K: окно сети берётся без пересборки поля на каждом шаге;
        # known, mem, visits - виды на середину (без полей)
        k = F.K
        self.padded = (np.zeros((n, hmax + 2 * k, wmax + 2 * k), bool),
                       np.zeros((n, hmax + 2 * k, wmax + 2 * k), np.uint8),
                       np.zeros((n, hmax + 2 * k, wmax + 2 * k), np.int32))
        self.known, self.mem, self.visits = (a[:, k:k + hmax, k:k + wmax] for a in self.padded)
        self.pos = np.zeros((n, 2), np.int64)
        self.exit = np.zeros((n, 2), np.int64)
        self.hw = np.zeros((n, 2), np.int64)
        self.last = np.full(n, -1, np.int64)
        self.steps = np.zeros(n, np.int64)
        self.cap = np.zeros(n, np.int64)
        self.shortest = np.zeros(n, np.int64)
        self.active = np.zeros(n, bool)
        self.ids = [None] * n

    def load(self, i, maze, cap, ident=None):
        h, w = maze.h, maze.w
        self.true[i] = 0
        self.true[i, :h, :w] = maze.cells
        self.known[i] = False
        self.mem[i] = 0
        self.visits[i] = 0
        self.pos[i] = maze.start
        self.exit[i] = maze.exit
        self.hw[i] = (h, w)
        self.last[i] = -1
        self.steps[i] = 0
        self.cap[i] = cap
        self.shortest[i] = maze.shortest()
        self.visits[i, maze.start[0], maze.start[1]] = 1
        self.active[i] = True
        self.ids[i] = ident
        self._see(np.array([i]))

    def _see(self, idx):
        if len(idx) == 0:
            return
        r, c = self.pos[idx, 0], self.pos[idx, 1]
        h, w = self.hw[idx, 0], self.hw[idx, 1]
        for dr in range(-VIEW, VIEW + 1):
            for dc in range(-VIEW, VIEW + 1):
                rr, cc = r + dr, c + dc
                ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                b, rr, cc = idx[ok], rr[ok], cc[ok]
                self.known[b, rr, cc] = True
                self.mem[b, rr, cc] = self.true[b, rr, cc]

    def encode(self, idx=None):
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        return F.encode(None, None, None, self.pos[idx], self.exit[idx], self.hw[idx], self.last[idx],
                        padded=self.padded, rows=idx)

    def distance_to_exit(self, idx=None):
        idx = np.arange(self.n) if idx is None else idx
        return np.abs(self.pos[idx] - self.exit[idx]).sum(axis=1)

    def step(self, actions):
        """Шаг всех активных (actions - настоящие направления). Ход в стену - ошибка:
        маска ходов не даёт его выбрать. Возвращает (награда, дошёл, упёрся в предел)."""
        idx = np.nonzero(self.active)[0]
        a = np.asarray(actions)[idx]
        here = self.true[idx, self.pos[idx, 0], self.pos[idx, 1]]
        bits = np.array([1, 2, 4, 8])[a]
        if np.any((here & bits) == 0):
            raise RuntimeError("шаг в стену")
        before = self.distance_to_exit(idx)
        self.pos[idx, 0] += np.array([-1, 0, 1, 0])[a]
        self.pos[idx, 1] += np.array([0, 1, 0, -1])[a]
        self.last[idx] = a
        self.steps[idx] += 1
        v = self.visits[idx, self.pos[idx, 0], self.pos[idx, 1]]
        self.visits[idx, self.pos[idx, 0], self.pos[idx, 1]] = v + 1
        self._see(idx)
        after = self.distance_to_exit(idx)
        reward = np.zeros(self.n, np.float32)
        reward[idx] = -STEP_COST + self.shaping * (before - after) + self.new_cell * (v == 0)
        done = np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        done[idx] = np.all(self.pos[idx] == self.exit[idx], axis=1)
        trunc[idx] = ~done[idx] & (self.steps[idx] >= self.cap[idx])
        return reward, done, trunc


def random_maze(rng, sizes=(5, 20), kinds=TRAIN_KINDS, loops=(0.0, 0.0, 0.25, 0.5)):
    """Случайный лабиринт для обучения: генератор, размер, петли, углы входа и выхода."""
    kind = kinds[rng.integers(len(kinds))]
    h = int(rng.integers(sizes[0], sizes[1] + 1))
    w = int(rng.integers(sizes[0], sizes[1] + 1))
    m = generate(kind, h, w, int(rng.integers(1 << 30)), loops=float(loops[rng.integers(len(loops))]))
    return m.transformed(bool(rng.integers(2)), bool(rng.integers(2)))


def run_policy(choose, mazes, cap_mult=10, batch=128):
    """Пройти список лабиринтов пачками. choose(env, idx) -> настоящие направления для
    активных idx. Возвращает [(шагов, дошёл)] по порядку."""
    out = [None] * len(mazes)
    hm = max(m.h for m in mazes)
    wm = max(m.w for m in mazes)
    for s in range(0, len(mazes), batch):
        part = mazes[s:s + batch]
        env = BatchEnv(len(part), hm, wm)
        for i, m in enumerate(part):
            env.load(i, m, cap_mult * m.h * m.w, ident=s + i)
        while env.active.any():
            idx = np.nonzero(env.active)[0]
            acts = np.zeros(env.n, np.int64)
            acts[idx] = choose(env, idx)
            _, done, trunc = env.step(acts)
            for i in np.nonzero(done | trunc)[0]:
                out[env.ids[i]] = (int(env.steps[i]), bool(done[i]))
                env.active[i] = False
    return out


def net_chooser(net, greedy=True, rng=None):
    """choose для run_policy по сети: самый вероятный ход или случайный по вероятностям."""
    from aikit.nn.losses import softmax

    def choose(env, idx):
        x, flips, legal = env.encode(idx)
        logits, _ = net.forward(x, legal)
        if greedy:
            a = logits.argmax(axis=1)
        else:
            p = softmax(logits)
            u = rng.random(len(idx))[:, None]
            a = (p.cumsum(axis=1) < u).sum(axis=1)
            a = np.minimum(a, 3)
            bad = ~legal[np.arange(len(a)), a]          # край округления суммы
            a[bad] = logits[bad].argmax(axis=1)
        return F.to_real(a, flips)
    return choose
