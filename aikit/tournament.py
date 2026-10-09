"""Турнир для паспорта модели: N партий, у каждой своё зерно, итог партия за партией.

Партия i получает генератор numpy.random.default_rng([seed, i]) и цвет: чётные партии
проверяемый игрок начинает, нечётные - соперник. Поэтому любую партию можно пересчитать
отдельно и сверить с паспортом, а партии - раздать по процессам.

play_one(i, rng, subject_first) -> 'W' | 'D' | 'L' с точки зрения проверяемого игрока.
"""
import numpy as np


def game_rng(seed, i):
    return np.random.default_rng([int(seed), int(i)])


def run(play_one, games, seed, indices=None, processes=1, pool=None):
    """Строка исходов 'WDL...' длиной games (или по indices - для выборки в тестах)."""
    idx = list(range(games)) if indices is None else list(indices)
    jobs = [(play_one, i, seed) for i in idx]
    if pool is not None:
        out = pool.map(_job, jobs)
    elif processes > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes) as p:
            out = p.map(_job, jobs)
    else:
        out = [_job(j) for j in jobs]
    return "".join(out)


def _job(job):
    play_one, i, seed = job
    r = play_one(i, game_rng(seed, i), i % 2 == 0)
    if r not in ("W", "D", "L"):
        raise ValueError(f"партия {i}: исход {r!r}")
    return r


def summary(results):
    """{'games', 'wins', 'draws', 'losses', 'score', 'results'}; score - (победы + ничьи/2) / партии."""
    w, d, l = results.count("W"), results.count("D"), results.count("L")
    n = max(len(results), 1)
    return {"games": len(results), "wins": w, "draws": d, "losses": l,
            "score": round((w + d / 2) / n, 4), "results": results}
