"""Закон честности (spec §3): правка закрытой части лабиринта не меняет ходов ИИ, пока
он её не увидел. ИИ получает только память игрока (fog.Memory)."""
import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from game.ai.agents import NetAgent, Optimist, RightHand, Tremaux
from game.ai.net import PolicyNet
from game.core.fog import Run
from game.core.generators import KINDS, generate
from game.core.maze import DC, DR, Maze

_NET = PolicyNet(seed=7)


def agents():
    return [RightHand(), Tremaux(3), Optimist(), NetAgent(_NET, "net", greedy=True),
            NetAgent(_NET, "net_sampled", greedy=False, seed=5)]


def trace(agent, maze, limit):
    """Ходы и, после каждого хода, какие клетки уже видны."""
    run = Run(maze)
    agent.reset(run.memory)
    moves, seen = [], [run.memory.known.copy()]
    while not run.done and len(moves) < limit:
        d = agent.act(run.memory)
        assert run.step(d)
        moves.append(d)
        seen.append(run.memory.known.copy())
    return moves, seen


def scramble_unseen(maze, seen, rng):
    """Случайно переставить стены между клетками, которых игрок не видел: проход меняется,
    только если обе его клетки невидимы."""
    cells = maze.cells.copy()
    out = Maze(cells, maze.start, maze.exit)
    changed = 0
    for r in range(maze.h):
        for c in range(maze.w):
            for d in (1, 2):
                r2, c2 = r + DR[d], c + DC[d]
                if r2 < maze.h and c2 < maze.w and not seen[r, c] and not seen[r2, c2]:
                    if rng.random() < 0.5:
                        if out.is_open(r, c, d):
                            out.wall(r, c, d)
                        else:
                            out.carve(r, c, d)
                        changed += 1
    return out, changed


@pytest.mark.parametrize("idx", range(5))
@given(kind=st.sampled_from(KINDS), n=st.integers(6, 14), seed=st.integers(0, 10 ** 6),
       cut=st.integers(0, 60))
@settings(max_examples=60)
def test_unseen_part_does_not_change_moves(idx, kind, n, seed, cut):
    maze = generate(kind, n, n, seed, loops=0.2)
    moves, seen = trace(agents()[idx], maze, limit=cut)
    if not moves:
        return
    other, _ = scramble_unseen(maze, seen[-1], random.Random(seed))
    moves2, _ = trace(agents()[idx], other, limit=len(moves))
    assert moves2 == moves


def test_scramble_really_changes_the_maze():
    maze = generate("dfs", 12, 12, 1)
    _, seen = trace(RightHand(), maze, limit=5)
    other, changed = scramble_unseen(maze, seen[-1], random.Random(1))
    assert changed > 50 and not np.array_equal(other.cells, maze.cells)


def test_seen_part_does_change_moves_somewhere():
    """Обратная проверка: если правка видна, ход хотя бы где-то другой - тест выше не пустой."""
    differ = 0
    for s in range(20):
        maze = generate("kruskal", 10, 10, s)
        moves, _ = trace(Optimist(), maze, limit=200)
        other = generate("kruskal", 10, 10, s + 1000)
        moves2, _ = trace(Optimist(), other, limit=200)
        differ += moves != moves2
    assert differ > 10


def test_memory_holds_nothing_unseen():
    maze = generate("prim", 10, 10, 3)
    run = Run(maze)
    for d in trace(RightHand(), maze, 25)[0]:
        run.step(d)
        mem = run.memory
        assert np.all(mem.cells[~mem.known] == 0)
