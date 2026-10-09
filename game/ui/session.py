"""Что идёт на экране, без Qt: проход человека, гонка, просмотр агентов.

Участник (Runner) - проход одного лабиринта: свой Run (туман и память) и, если это ИИ,
свой агент. Агент получает только память; читерам (BFS, A*) лабиринт дают явно.
"""
import random
import time

from game.ai.agents import level_agent, make_agent
from game.core.fog import Run
from game.core.generators import KINDS, generate

SIZES = {"s": 8, "m": 12, "l": 16, "xl": 24}
LOOPS = 0.3


def make_maze(size, kind, loops, seed):
    """Лабиринт по настройкам окна. kind 'random' - генератор по зерну."""
    n = SIZES[size]
    rng = random.Random(seed)
    if kind == "random":
        kind = rng.choice(KINDS)
    maze = generate(kind, n, n, seed, loops=LOOPS if loops else 0.0)
    return maze.transformed(bool(rng.getrandbits(1)), bool(rng.getrandbits(1)))


class Runner:
    def __init__(self, maze, agent=None, name="you", missing=False):
        self.maze = maze
        self.missing = missing          # агента нет (сеть не обучена): участник не ходит
        self.run = Run(maze.copy())
        self.agent = agent
        self.name = name
        self.limit = 10 * maze.h * maze.w
        self.finish_order = None
        if agent is not None:
            if getattr(agent, "cheater", False):
                agent.reset(self.run.memory, self.run.reveal())
            else:
                agent.reset(self.run.memory)

    @property
    def memory(self):
        return self.run.memory

    @property
    def done(self):
        return self.run.done

    @property
    def failed(self):
        return not self.run.done and self.memory.steps >= self.limit

    @property
    def stopped(self):
        return self.missing or self.done or self.failed

    def ai_step(self):
        if self.stopped:
            return False
        return self.run.step(self.agent.act(self.memory))


class Session:
    """mode: 'solo', 'race', 'watch'. Человек - runners[0] в solo и race."""

    def __init__(self, mode, maze, level="strong", agents=(), seed=0):
        self.mode = mode
        self.maze = maze
        self.level = level
        self.seed = seed
        self.runners = []
        self.started = None
        self.ended = None
        self.recorded = False
        self.places = 0
        if mode in ("solo", "race"):
            self.runners.append(Runner(maze, None, "you"))
        if mode == "race":
            self.runners.append(Runner(maze, level_agent(level, seed), "ai"))
        if mode == "watch":
            for name in agents:
                agent = make_agent(name, seed)
                self.runners.append(Runner(maze, agent, name, missing=agent is None))

    @property
    def human(self):
        return self.runners[0] if self.mode in ("solo", "race") else None

    @property
    def ai_runners(self):
        return [r for r in self.runners if r.agent is not None and not r.missing]

    def elapsed(self, now=None):
        if self.started is None:
            return 0.0
        return (self.ended or now or time.monotonic()) - self.started

    def move(self, d):
        """Ход человека. Возвращает True, если шагнул."""
        h = self.human
        if h is None or self.over:
            return False
        if self.started is None:
            self.started = time.monotonic()
        ok = h.run.step(d)
        self._check()
        return ok

    def tick(self):
        """Шаг всех ИИ (в гонке - после первого хода человека)."""
        if self.over:
            return False
        if self.mode == "race" and self.started is None:
            return False
        if self.started is None:
            self.started = time.monotonic()
        moved = False
        for r in self.ai_runners:
            moved |= r.ai_step()
        self._check()
        return moved

    def _check(self):
        for r in self.runners:
            if r.stopped and r.finish_order is None:
                if r.done:
                    self.places += 1
                    r.finish_order = self.places
                else:
                    r.finish_order = 0
        if self.over and self.ended is None:
            self.ended = time.monotonic()

    @property
    def over(self):
        if self.mode == "solo":
            return self.human.done
        if self.mode == "race":
            return any(r.done for r in self.runners) or all(r.stopped for r in self.runners)
        return all(r.stopped for r in self.runners)

    def outcome(self):
        """Для гонки: 'win' или 'loss' с точки зрения человека; иначе None."""
        if self.mode != "race" or not self.over:
            return None
        return "win" if self.human.done else "loss"

    def stats_key(self):
        return f"race/{self.level}"
