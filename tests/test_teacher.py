"""Оптимист пачкой (учитель для подражания) выбирает ход в ход как одиночный Оптимист."""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from game.ai import teacher
from game.ai.agents import Optimist, optimistic_distances
from game.ai.env import BatchEnv, random_maze
from game.core.fog import Run


@given(st.integers(0, 10 ** 6))
@settings(max_examples=40)
def test_batch_teacher_equals_single_optimist(seed):
    rng = np.random.default_rng(seed)
    mazes = [random_maze(rng, sizes=(3, 12)) for _ in range(5)]
    env = BatchEnv(5, 12, 12)
    runs = [Run(m) for m in mazes]
    singles = [Optimist() for _ in mazes]
    for i, m in enumerate(mazes):
        env.load(i, m, 10 ** 4)
        singles[i].reset(runs[i].memory)
    for _ in range(60):
        idx = np.nonzero(env.active)[0]
        if len(idx) == 0:
            break
        acts = teacher.choose(env.known[idx], env.mem[idx], env.visits[idx], env.pos[idx], env.exit[idx],
                              env.hw[idx])
        full = np.zeros(5, np.int64)
        full[idx] = acts
        for k, i in enumerate(idx):
            mem = runs[i].memory
            want = singles[i].act(mem)
            assert acts[k] == want
            h, w = mazes[i].size
            d = teacher.distances(env.known[i:i + 1], env.mem[i:i + 1], env.hw[i:i + 1], env.exit[i:i + 1])[0]
            opt = optimistic_distances(mem)
            assert np.array_equal(np.where(d[:h, :w] >= teacher.BIG, -1, d[:h, :w]), opt)
            runs[i].step(want)
        _, done, trunc = env.step(full)
        env.active[done | trunc] = False
