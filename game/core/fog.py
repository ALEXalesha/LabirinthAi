"""Туман и память. Игрок и ИИ видят одинаково: клетки в квадрате VIEW вокруг себя
(при VIEW = 1 - 3 x 3) и всё, что видели раньше. Рамка лабиринта, вход и выход видны
с самого начала.

Run держит настоящий лабиринт и выдаёт наружу только Memory - то, что видно игроку.
ИИ получает Memory и больше ничего (закон честности в tests/test_honesty.py: правка
невиданной части лабиринта не меняет ходов ИИ, пока он её не увидел).
"""
import numpy as np

from game.core.maze import BIT, DC, DR, OPPOSITE

VIEW = 1


class Memory:
    """Карта, которую знает игрок: known - видел ли клетку, cells - проходы видимых
    клеток (у невиданных 0), visits - сколько раз стоял в клетке, pos - где он сейчас."""

    def __init__(self, h, w, start, exit):
        self.h, self.w = h, w
        self.known = np.zeros((h, w), bool)
        self.cells = np.zeros((h, w), np.uint8)
        self.visits = np.zeros((h, w), np.int32)
        self.start = tuple(start)
        self.exit = tuple(exit)
        self.pos = tuple(start)
        self.steps = 0
        self.last = -1              # направление последнего шага, -1 - ещё не ходил
        self.trail = [tuple(start)]

    def edge(self, r, c, d):
        """1 - проход виден, 0 - стена видна, -1 - неизвестно."""
        r2, c2 = r + DR[d], c + DC[d]
        if not (0 <= r2 < self.h and 0 <= c2 < self.w):
            return 0
        if self.known[r, c]:
            return 1 if self.cells[r, c] & BIT[d] else 0
        if self.known[r2, c2]:
            return 1 if self.cells[r2, c2] & BIT[OPPOSITE[d]] else 0
        return -1

    def can_move(self, d):
        return self.edge(*self.pos, d) == 1

    def moves(self):
        """Куда можно шагнуть из текущей клетки (её стены всегда видны)."""
        return [d for d in range(4) if self.can_move(d)]

    @property
    def done(self):
        return self.pos == self.exit

    def copy(self):
        m = Memory(self.h, self.w, self.start, self.exit)
        m.known = self.known.copy()
        m.cells = self.cells.copy()
        m.visits = self.visits.copy()
        m.pos, m.steps, m.last, m.trail = self.pos, self.steps, self.last, list(self.trail)
        return m


class Run:
    """Один проход лабиринта: настоящий лабиринт внутри, память - наружу."""

    def __init__(self, maze, view=VIEW):
        self._maze = maze
        self.view = view
        self.memory = Memory(maze.h, maze.w, maze.start, maze.exit)
        self.memory.visits[maze.start] = 1
        self._see()

    def _see(self):
        m = self.memory
        r, c = m.pos
        v = self.view
        r0, r1 = max(0, r - v), min(m.h, r + v + 1)
        c0, c1 = max(0, c - v), min(m.w, c + v + 1)
        m.known[r0:r1, c0:c1] = True
        m.cells[r0:r1, c0:c1] = self._maze.cells[r0:r1, c0:c1]

    def step(self, d):
        """Шаг в сторону d. Стена - стоим на месте, шаг не считается; вернёт False."""
        m = self.memory
        if m.done or not self._maze.is_open(*m.pos, d):
            return False
        m.pos = (m.pos[0] + DR[d], m.pos[1] + DC[d])
        m.steps += 1
        m.last = d
        m.visits[m.pos] += 1
        m.trail.append(m.pos)
        self._see()
        return True

    @property
    def done(self):
        return self.memory.done

    def reveal(self):
        """Весь лабиринт - только для конца игры и для «читерских» решателей."""
        return self._maze


def play(agent, maze, limit=None, view=VIEW):
    """Агент проходит лабиринт; ход за ходом он видит только Memory. Возвращает Run.
    limit - сколько шагов самое большее (по умолчанию 10 x клеток)."""
    run = Run(maze, view)
    limit = limit if limit is not None else 10 * maze.h * maze.w
    agent.reset(run.memory)
    while not run.done and run.memory.steps < limit:
        d = agent.act(run.memory)
        if not run.step(d):
            raise RuntimeError(f"{type(agent).__name__}: шаг в стену {d} из {run.memory.pos}")
    return run
