"""Законы генераторов (spec §2): связность, нет изолированных клеток, у идеальных
проходов на один меньше, чем клеток; проходы двусторонние; одно зерно - один лабиринт."""
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from game.core.generators import KINDS, generate
from game.core.maze import BIT, E, Maze, N, S, W

sizes = st.integers(2, 18)
seeds = st.integers(0, 10 ** 6)


@given(st.sampled_from(KINDS), sizes, sizes, seeds)
def test_every_generator_gives_perfect_maze(kind, h, w, seed):
    m = generate(kind, h, w, seed)
    assert m.symmetric(), m.to_text()
    assert m.connected(), m.to_text()
    assert np.all(m.cells != 0), "изолированная клетка"
    assert m.passages() == h * w - 1, m.to_text()
    assert m.shortest() > 0


@given(st.sampled_from(KINDS), sizes, sizes, seeds, st.floats(0.05, 1.0))
def test_loops_keep_maze_connected_and_add_passages(kind, h, w, seed, share):
    m = generate(kind, h, w, seed, loops=share)
    assert m.symmetric() and m.connected()
    assert m.passages() >= h * w - 1
    base = generate(kind, h, w, seed)
    assert np.all((base.cells & m.cells) == base.cells), "петли не сносят старых проходов"


def test_loops_really_add_cycles():
    m = generate("dfs", 15, 15, 3, loops=0.5)
    assert m.passages() > 15 * 15 - 1
    assert m.dead_ends() < generate("dfs", 15, 15, 3).dead_ends()


@given(st.sampled_from(KINDS), sizes, sizes, seeds)
def test_same_seed_same_maze(kind, h, w, seed):
    assert np.array_equal(generate(kind, h, w, seed).cells, generate(kind, h, w, seed).cells)


def test_different_seeds_give_different_mazes():
    for kind in KINDS:
        cells = {generate(kind, 12, 12, s).cells.tobytes() for s in range(20)}
        assert len(cells) == 20, kind


def test_generators_differ_in_character():
    """DFS - длинные коридоры (мало тупиков), Прим - много коротких тупиков."""
    dfs = np.mean([generate("dfs", 20, 20, s).dead_ends() for s in range(20)])
    prim = np.mean([generate("prim", 20, 20, s).dead_ends() for s in range(20)])
    assert prim > 1.5 * dfs


def test_binary_tree_has_open_top_row_and_left_column():
    m = generate("binary", 10, 10, 1)
    assert all(m.is_open(0, c, W) for c in range(1, 10))
    assert all(m.is_open(r, 0, N) for r in range(1, 10))


def test_carve_is_two_sided_and_cannot_leave_board():
    m = Maze.closed(3, 3)
    m.carve(1, 1, E)
    assert m.cells[1, 1] & BIT[E] and m.cells[1, 2] & BIT[W]
    m.carve(0, 0, S)
    assert m.symmetric()
    try:
        m.carve(0, 0, N)
    except ValueError:
        pass
    else:
        raise AssertionError("проход наружу")


def test_symmetric_catches_one_sided_passage():
    m = generate("dfs", 5, 5, 1)
    m.cells[2, 2] ^= BIT[S]
    assert not m.symmetric()


def test_transform_moves_start_exit_and_keeps_paths():
    m = generate("kruskal", 7, 11, 2)
    for v in (False, True):
        for h in (False, True):
            t = m.transformed(v, h)
            assert t.symmetric() and t.connected()
            assert t.shortest() == m.shortest()
            assert t.passages() == m.passages()


def test_shortest_path_matches_distance():
    m = generate("wilson", 9, 9, 4)
    path = m.shortest_path()
    assert path[0] == m.start and path[-1] == m.exit
    assert len(path) - 1 == m.shortest()
    for a, b in zip(path, path[1:]):
        assert any((r, c) == b for _, r, c in m.neighbors(*a))
