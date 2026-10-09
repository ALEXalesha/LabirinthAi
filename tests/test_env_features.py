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

@given(st.integers(0, 10 ** 6))
@settings(max_examples=60)
def test_fast_encode_equals_reference(seed):
    """Быстрое окно (сразу развёрнутое, без пересборки поля) совпадает с прежним."""
    rng = np.random.default_rng(seed)
    mazes = [random_maze(rng, sizes=(3, 12)) for _ in range(6)]
    env = BatchEnv(6, 12, 12)
    for i, m in enumerate(mazes):
        env.load(i, m, 1000)
    for _ in range(int(rng.integers(0, 40))):
        idx = np.nonzero(env.active)[0]
        acts = np.zeros(6, np.int64)
        for i in idx:
            here = env.mem[i, env.pos[i, 0], env.pos[i, 1]]
            acts[i] = rng.choice([d for d in range(4) if here & (1 << d)])
        _, done, trunc = env.step(acts)
        env.active[done | trunc] = False
    idx = np.arange(6)
    want = reference_encode(env.known.copy(), env.mem.copy(), env.visits.copy(), env.pos, env.exit,
                            env.hw, env.last)
    got = env.encode(idx)
    for a, b in zip(got, want):
        assert np.array_equal(a, b)
    part = np.array([4, 1])
    got_part = env.encode(part)
    assert np.array_equal(got_part[0], want[0][part])


def reference_encode(known, cells, visits, pos, exit_, hw, last):
    """Прежнее encode (всё поле в плоскости, потом окно и отражения) - образец для быстрого."""
    B, H, W = known.shape
    P = F.planes(known, cells, visits, hw, exit_)
    pad = np.zeros((B, H + 2 * F.K, W + 2 * F.K, F.C), np.float32)
    pad[:, F.K:F.K + H, F.K:F.K + W] = P
    off = np.arange(F.SIDE)
    rr = pos[:, 0, None] + off[None, :]               # в координатах с полями
    cc = pos[:, 1, None] + off[None, :]
    crop = pad[np.arange(B)[:, None, None], rr[:, :, None], cc[:, None, :]]   # (B, S, S, F.C)

    dy = exit_[:, 0] - pos[:, 0]
    dx = exit_[:, 1] - pos[:, 1]
    flip_v = dy < 0
    flip_h = dx < 0
    if flip_v.any():
        v = crop[flip_v][:, ::-1]
        crop[flip_v] = v[..., [0, 1, 4, 3, 2, 5, 6, 7, 8]]
    if flip_h.any():
        h = crop[flip_h][:, :, ::-1]
        crop[flip_h] = h[..., [0, 1, 2, 5, 4, 3, 6, 7, 8]]
    flips = np.stack([flip_v, flip_h], axis=1)

    g = np.zeros((B, F.GLOBAL), np.float32)
    g[:, 0] = np.minimum(np.abs(dy) / 16.0, 3.0)
    g[:, 1] = np.minimum(np.abs(dx) / 16.0, 3.0)
    g[:, 2] = dy == 0
    g[:, 3] = dx == 0
    has_last = last >= 0
    if has_last.any():
        lc = F.from_real(np.where(has_last, last, 0), flips)
        g[np.arange(B)[has_last], 4 + lc[has_last]] = 1.0

    here = cells[np.arange(B), pos[:, 0], pos[:, 1]]
    legal_real = np.stack([(here & b) > 0 for b in (1, 2, 4, 8)], axis=1)
    legal = np.zeros_like(legal_real)
    for d in range(4):
        legal[np.arange(B), F.from_real(np.full(B, d), flips)] = legal_real[:, d]
    x = np.concatenate([crop.reshape(B, -1), g], axis=1)
    return x, flips, legal


def test_encode_ignores_walls_of_unseen_cells():
    """Даже если в cells попали проходы невиданных клеток, сеть их не получает."""
    m = generate("prim", 9, 9, 7)
    mem = Run(m).memory
    full = m.cells[None].copy()
    a = F.encode(mem.known[None], full, mem.visits[None], np.array([mem.pos]), np.array([mem.exit]),
                 np.array([[9, 9]]), np.array([-1]))
    b = F.encode_memory(mem)
    assert np.array_equal(a[0], b[0])
