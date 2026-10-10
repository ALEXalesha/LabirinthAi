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


def test_hint_marks_optimist_distances_in_window_directions():
    from game.ai import features as F
    rng = np.random.default_rng(5)
    for _ in range(30):
        m = random_maze(rng, sizes=(4, 12))
        run = Run(m)
        opt = Optimist()
        opt.reset(run.memory)
        for _ in range(int(rng.integers(0, 15))):
            if run.done:
                break
            run.step(opt.act(run.memory))
        if run.done:
            continue
        mem = run.memory
        x, flips, legal = F.encode_memory(mem, hints=True)
        h = x[0, F.SIZE:]
        assert x.shape[1] == F.SIZE_HINT
        dist = optimistic_distances(mem)
        best = min(dist[mem.pos[0] + dr, mem.pos[1] + dc] for d, (dr, dc) in
                   enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))) if d in mem.moves())
        for d, (dr, dc) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            c = int(F.from_real(np.array([d]), flips)[0])
            assert h[3 * c] == (d in mem.moves()) == legal[0, c]
            if d in mem.moves():
                nd = dist[mem.pos[0] + dr, mem.pos[1] + dc]
                assert h[3 * c + 2] == (nd == best)
                assert abs(h[3 * c + 1] - min((nd - best) / 2, 1)) < 1e-6
        assert abs(h[12] - min(best / 32, 3)) < 1e-6
