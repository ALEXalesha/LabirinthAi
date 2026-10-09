"""Лабиринт: прямоугольник H x W клеток, проходы между соседями.

Клетка хранит битовую маску открытых сторон: N=1, E=2, S=4, W=8. Проход всегда
двусторонний: если у клетки открыт восток, у соседа справа открыт запад (закон
symmetric() в тестах). Наружная рамка закрыта всегда; вход и выход - клетки внутри
(start, exit), их видно на поле с самого начала, как рамку лабиринта.

Только Python и numpy, без Qt.
"""
from collections import deque

import numpy as np

N, E, S, W = 0, 1, 2, 3
DIRS = (N, E, S, W)
BIT = (1, 2, 4, 8)
DR = (-1, 0, 1, 0)
DC = (0, 1, 0, -1)
OPPOSITE = (S, W, N, E)
NAMES = ("N", "E", "S", "W")


class Maze:
    """cells - numpy (H, W) uint8 маски проходов; start, exit - (строка, столбец)."""

    def __init__(self, cells, start=None, exit=None, kind="", seed=None):
        self.cells = np.asarray(cells, dtype=np.uint8)
        h, w = self.cells.shape
        self.start = tuple(start) if start is not None else (0, 0)
        self.exit = tuple(exit) if exit is not None else (h - 1, w - 1)
        self.kind = kind
        self.seed = seed

    @classmethod
    def closed(cls, h, w, **kw):
        return cls(np.zeros((h, w), np.uint8), **kw)

    @property
    def h(self):
        return self.cells.shape[0]

    @property
    def w(self):
        return self.cells.shape[1]

    @property
    def size(self):
        return self.cells.shape

    def inside(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w

    def is_open(self, r, c, d):
        return bool(self.cells[r, c] & BIT[d])

    def carve(self, r, c, d):
        """Открыть проход из (r, c) в сторону d - сразу с обеих сторон."""
        r2, c2 = r + DR[d], c + DC[d]
        if not self.inside(r2, c2):
            raise ValueError(f"проход наружу из {(r, c)} в {NAMES[d]}")
        self.cells[r, c] |= BIT[d]
        self.cells[r2, c2] |= BIT[OPPOSITE[d]]

    def wall(self, r, c, d):
        r2, c2 = r + DR[d], c + DC[d]
        self.cells[r, c] &= ~np.uint8(BIT[d])
        if self.inside(r2, c2):
            self.cells[r2, c2] &= ~np.uint8(BIT[OPPOSITE[d]])

    def neighbors(self, r, c):
        """Соседи, куда есть проход: [(направление, строка, столбец)]."""
        m = int(self.cells[r, c])
        return [(d, r + DR[d], c + DC[d]) for d in DIRS if m & BIT[d]]

    def passages(self):
        """Число проходов (каждый считается один раз)."""
        east = int(np.count_nonzero(self.cells[:, :-1] & BIT[E]))
        south = int(np.count_nonzero(self.cells[:-1, :] & BIT[S]))
        return east + south

    def symmetric(self):
        """Каждый проход открыт с обеих сторон, наружу проходов нет."""
        c = self.cells
        if np.any(c[0, :] & BIT[N]) or np.any(c[-1, :] & BIT[S]):
            return False
        if np.any(c[:, 0] & BIT[W]) or np.any(c[:, -1] & BIT[E]):
            return False
        if np.any(((c[:, :-1] & BIT[E]) > 0) != ((c[:, 1:] & BIT[W]) > 0)):
            return False
        if np.any(((c[:-1, :] & BIT[S]) > 0) != ((c[1:, :] & BIT[N]) > 0)):
            return False
        return True

    def distances(self, src=None):
        """Расстояние в шагах от src (по умолчанию от выхода) до каждой клетки; -1 - не дойти."""
        src = self.exit if src is None else tuple(src)
        dist = np.full(self.cells.shape, -1, np.int32)
        dist[src] = 0
        q = deque([src])
        cells = self.cells
        while q:
            r, c = q.popleft()
            m = int(cells[r, c])
            nd = dist[r, c] + 1
            for d in DIRS:
                if m & BIT[d]:
                    r2, c2 = r + DR[d], c + DC[d]
                    if dist[r2, c2] < 0:
                        dist[r2, c2] = nd
                        q.append((r2, c2))
        return dist

    def connected(self):
        return bool(np.all(self.distances(self.start) >= 0))

    def shortest(self):
        """Длина кратчайшего пути от входа до выхода (шагов)."""
        return int(self.distances()[self.start])

    def shortest_path(self):
        """Клетки кратчайшего пути от входа до выхода включительно."""
        dist = self.distances()
        if dist[self.start] < 0:
            return []
        path = [self.start]
        r, c = self.start
        while (r, c) != self.exit:
            for d, r2, c2 in self.neighbors(r, c):
                if dist[r2, c2] == dist[r, c] - 1:
                    r, c = r2, c2
                    break
            path.append((r, c))
        return path

    def dead_ends(self):
        """Клетки с одним проходом."""
        bits = np.unpackbits(self.cells[..., None], axis=-1)[..., 4:].sum(axis=-1)
        return int(np.count_nonzero(bits == 1))

    def copy(self):
        return Maze(self.cells.copy(), self.start, self.exit, self.kind, self.seed)

    def transformed(self, flip_v=False, flip_h=False):
        """Отражение по вертикали (строки наоборот) и по горизонтали - для разнообразия
        при обучении: вход и выход переезжают вместе с проходами."""
        cells = self.cells.copy()
        start, exit_ = list(self.start), list(self.exit)
        if flip_v:
            cells = cells[::-1].copy()
            n, s = (cells & BIT[N]) > 0, (cells & BIT[S]) > 0
            cells &= ~np.uint8(BIT[N] | BIT[S])
            cells |= np.where(n, BIT[S], 0).astype(np.uint8) | np.where(s, BIT[N], 0).astype(np.uint8)
            start[0], exit_[0] = self.h - 1 - start[0], self.h - 1 - exit_[0]
        if flip_h:
            cells = cells[:, ::-1].copy()
            e, w = (cells & BIT[E]) > 0, (cells & BIT[W]) > 0
            cells &= ~np.uint8(BIT[E] | BIT[W])
            cells |= np.where(e, BIT[W], 0).astype(np.uint8) | np.where(w, BIT[E], 0).astype(np.uint8)
            start[1], exit_[1] = self.w - 1 - start[1], self.w - 1 - exit_[1]
        return Maze(cells, tuple(start), tuple(exit_), self.kind, self.seed)

    def to_text(self):
        """Лабиринт символами - для отладки и сообщений тестов."""
        lines = ["+" + "--+" * self.w]
        for r in range(self.h):
            row, bottom = "|", "+"
            for c in range(self.w):
                mark = "S " if (r, c) == self.start else ("X " if (r, c) == self.exit else "  ")
                row += mark + (" " if self.is_open(r, c, E) else "|")
                bottom += ("  " if self.is_open(r, c, S) else "--") + "+"
            lines += [row, bottom]
        return "\n".join(lines)
