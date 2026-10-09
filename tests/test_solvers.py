"""Туман и классические решатели: все доходят до выхода, читеры идут кратчайшим путём,
честные никогда не шагают в стену."""
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from game.ai.agents import Cheater, Optimist, RightHand, Tremaux, optimistic_distances, search
from game.core.fog import Run, play
from game.core.generators import KINDS, generate
from game.core.maze import E, S


def test_fog_shows_only_3x3_around_and_remembers():
    m = generate("dfs", 8, 8, 1)
    run = Run(m)
    mem = run.memory
    assert mem.known.sum() == 4                       # угол: 2 x 2
    assert mem.known[:2, :2].all()
    d = mem.moves()[0]
    run.step(d)
    assert mem.known.sum() == 6
    assert np.all(mem.cells[~mem.known] == 0)
    assert np.array_equal(mem.cells[mem.known], m.cells[mem.known])


def test_step_into_wall_does_not_count():
    m = generate("dfs", 6, 6, 2)
    run = Run(m)
    walls = [d for d in range(4) if not m.is_open(0, 0, d)]
    assert not run.step(walls[0])
    assert run.memory.steps == 0 and run.memory.pos == (0, 0)


def test_edge_knowledge_from_neighbour():
    m = generate("dfs", 6, 6, 2)
    run = Run(m)
    mem = run.memory
    # клетка (2, 0) не видна, но её северная стена - это южная стена видимой (1, 0)
    assert not mem.known[2, 0]
    assert mem.edge(2, 0, 0) == (1 if m.is_open(1, 0, S) else 0)
    assert mem.edge(3, 3, E) == -1


@pytest.mark.parametrize("agent", [RightHand(), Tremaux(1), Optimist()], ids=lambda a: a.name)
@given(kind=st.sampled_from(KINDS), n=st.integers(2, 14), seed=st.integers(0, 10 ** 6),
       loops=st.sampled_from([0.0, 0.3]))
def test_honest_solvers_reach_exit(agent, kind, n, seed, loops):
    m = generate(kind, n, n + 1, seed, loops=loops)
    run = play(agent, m)
    assert run.done, (agent.name, m.to_text())
    assert run.memory.steps >= m.shortest()


@given(st.sampled_from(KINDS), st.integers(2, 14), st.integers(0, 10 ** 6))
def test_tremaux_walks_each_passage_at_most_twice(kind, n, seed):
    m = generate(kind, n, n, seed, loops=0.3)
    agent = Tremaux(seed)
    play(agent, m)
    assert max(agent.marks.values(), default=0) <= 2


@given(st.sampled_from(KINDS), st.integers(2, 14), st.integers(0, 10 ** 6))
def test_right_hand_on_perfect_maze_at_most_twice_every_passage(kind, n, seed):
    m = generate(kind, n, n, seed)
    run = play(RightHand(), m)
    assert run.memory.steps <= 2 * m.passages()


@pytest.mark.parametrize("kind", ["bfs", "astar"])
@given(st.sampled_from(KINDS), st.integers(2, 14), st.integers(0, 10 ** 6))
def test_cheaters_walk_shortest_path(kind, maze_kind, n, seed):
    m = generate(maze_kind, n, n, seed, loops=0.3)
    agent = Cheater(kind)
    run = Run(m)
    agent.reset(run.memory, m)
    while not run.done:
        assert run.step(agent.act(run.memory))
    assert run.memory.steps == m.shortest()


def test_astar_explores_no_more_than_bfs():
    tot_a = tot_b = 0
    for s in range(30):
        m = generate("kruskal", 15, 15, s, loops=0.3)
        tot_a += len(search(m, "astar")[1])
        tot_b += len(search(m, "bfs")[1])
    assert tot_a < tot_b


def test_cheater_refuses_without_maze():
    with pytest.raises(ValueError):
        Cheater().reset(Run(generate("dfs", 4, 4, 1)).memory)


def test_optimistic_distance_is_lower_bound_of_true_distance():
    m = generate("dfs", 10, 10, 5)
    run = Run(m)
    agent = Optimist()
    agent.reset(run.memory)
    true = m.distances()
    for _ in range(30):
        opt = optimistic_distances(run.memory)
        assert np.all(opt <= true)
        run.step(agent.act(run.memory))
        if run.done:
            break


def test_optimist_beats_right_hand_on_average():
    """Оптимист знает, где выход, - в среднем короче правой руки."""
    opt = rh = 0
    for s in range(40):
        m = generate("wilson", 12, 12, s)
        opt += play(Optimist(), m).memory.steps
        rh += play(RightHand(), m).memory.steps
    assert opt < rh
