"""Пачка лабиринтов (обучение) и одиночный проход (игра) видят одно и то же; окно сети
разворачивается правильно."""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from game.ai import features as F
from game.ai.agents import NetAgent, RightHand
from game.ai.env import BatchEnv, net_chooser, random_maze, run_policy
from game.ai.net import PolicyNet
from game.core.fog import Run, play
from game.core.generators import generate
from game.core.maze import DC, DR


@given(st.integers(0, 10 ** 6))
@settings(max_examples=40)
def test_batch_memory_equals_single_run(seed):
    rng = np.random.default_rng(seed)
    mazes = [random_maze(rng, sizes=(4, 10)) for _ in range(5)]
    env = BatchEnv(5, 10, 10)
    runs = [Run(m) for m in mazes]
    agents = [RightHand() for _ in mazes]
    for i, m in enumerate(mazes):
        env.load(i, m, 1000)
        agents[i].reset(runs[i].memory)
    for _ in range(30):
        acts = np.zeros(5, np.int64)
        for i, run in enumerate(runs):
            if env.active[i]:
                acts[i] = agents[i].act(run.memory)
        env.step(acts)
        for i, run in enumerate(runs):
            if not env.active[i]:
                continue
            run.step(int(acts[i]))
            mem = run.memory
            h, w = mazes[i].size
            assert np.array_equal(env.known[i, :h, :w], mem.known)
            assert np.array_equal(env.mem[i, :h, :w], mem.cells)
            assert np.array_equal(env.visits[i, :h, :w], mem.visits)
            assert tuple(env.pos[i]) == mem.pos
            x1 = env.encode(np.array([i]))[0]
            x2 = F.encode_memory(mem)[0]
            assert np.array_equal(x1, x2)
            if run.done:
                env.active[i] = False


def test_window_rotates_exit_to_bottom_right():
    m = generate("dfs", 9, 9, 1)
    for v in (False, True):
        for h in (False, True):
            t = m.transformed(v, h)
            mem = Run(t).memory
            x, flips, legal = F.encode_memory(mem)
            crop = x[0, :F.SIDE * F.SIDE * F.C].reshape(F.SIDE, F.SIDE, F.C)
            assert x[0, F.SIDE * F.SIDE * F.C + 0] > 0 and x[0, F.SIDE * F.SIDE * F.C + 1] > 0
            # из угла вход: поле внутри только снизу справа от центра окна
            assert crop[F.K, F.K, 0] == 1
            assert crop[F.K - 1, F.K, 0] == 0 and crop[F.K, F.K - 1, 0] == 0
            assert crop[F.K + 1, F.K, 0] == 1 and crop[F.K, F.K + 1, 0] == 1
            # разрешённые ходы в окне - это настоящие ходы, переведённые в окно
            real = set(mem.moves())
            canon = {int(F.from_real(np.array([d]), flips)[0]) for d in real}
            assert set(np.nonzero(legal[0])[0]) == canon
            # проход в окне смотрит туда же, куда настоящий
            for d in real:
                cd = int(F.from_real(np.array([d]), flips)[0])
                assert crop[F.K, F.K, 2 + cd] == 1


def test_direction_maps_are_involutions():
    for fv in (False, True):
        for fh in (False, True):
            flips = np.array([[fv, fh]] * 4)
            d = np.arange(4)
            assert np.array_equal(F.to_real(F.from_real(d, flips), flips), d)


def test_single_net_agent_matches_batch_runner():
    net = PolicyNet(seed=3)
    mazes = [generate(k, 8, 9, s) for s, k in enumerate(["dfs", "prim", "kruskal", "wilson"] * 3)]
    batch = run_policy(net_chooser(net, greedy=True), mazes, cap_mult=3)
    for m, (steps, ok) in zip(mazes, batch):
        run = play(NetAgent(net), m, limit=3 * m.h * m.w)
        assert (run.memory.steps, run.done) == (steps, ok)


def test_new_cell_is_cheaper():
    m = generate("dfs", 6, 6, 1)
    env = BatchEnv(1, 6, 6)
    env.load(0, m, 100)
    d = [d for d in range(4) if m.is_open(0, 0, d)][0]
    r, done, trunc = env.step(np.array([d]))
    assert r[0] < 0 and not done[0]
    assert abs(r[0] - (-0.01 + 0.005)) < 1e-6        # шаг в новую клетку: дешевле
