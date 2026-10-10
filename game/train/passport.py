"""Паспорт модели: проверки силы (spec §4) и сведения об обучении.

    python -m game.train.passport                 # 1000 лабиринтов на проверку, 2 процесса
    python -m game.train.passport --games 100

Каждая проверка - незнакомые лабиринты одного размера: зёрна вне диапазона обучения
(там < 2^30) и вне кривой обучения (3e9...). Генераторы обучения (DFS, Прим, Краскал,
Уилсон) на 8, 12, 16 и 24 (24 больше всего, что сеть видела), плюс двоичное дерево и
комнаты - генераторы, которых сеть не видела совсем. Каждый третий лабиринт - с петлями.

Все агенты проходят одни и те же лабиринты через fog.play и видят только память. Сеть
выбирает ход случайно по своим вероятностям с зерном лабиринта. Первые SAMPLE лабиринтов
каждой проверки записываются поштучно: тест пересчитывает их и сверяет.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

SAMPLE = 10
SEED = 4_000_000_000
AGENTS = ("right_hand", "tremaux", "optimist", "net_ppo", "net_strong")
TRAIN = ("dfs", "prim", "kruskal", "wilson")
# (размер, генераторы: 'train' - те, что видела сеть, или один незнакомый)
CHECKS = [(8, "train"), (12, "train"), (16, "train"), (24, "train"), (16, "binary"), (16, "rooms")]


def check_maze(size, kind, i):
    from game.core.generators import generate
    k = TRAIN[i % len(TRAIN)] if kind == "train" else kind
    seed = SEED + 1_000_000 * size + 10_000 * (CHECKS.index((size, kind))) + i
    m = generate(k, size, size, seed, loops=0.25 if i % 3 == 0 else 0.0)
    return m.transformed(bool(i & 1), bool(i & 2))


def run_one(agent_name, size, kind, i):
    """Шаги агента и дошёл ли он в лабиринте i проверки (size, kind)."""
    from game.ai.agents import make_agent
    from game.core.fog import play
    m = check_maze(size, kind, i)
    agent = make_agent(agent_name, seed=SEED + i)
    if agent is None:
        return None
    run = play(agent, m)
    return run.memory.steps, bool(run.done)


def _part(job):
    agent_name, size, kind, indices = job
    out = []
    for i in indices:
        m = check_maze(size, kind, i)
        r = run_one(agent_name, size, kind, i)
        out.append((i, m.shortest(), r))
    return out


def run_check(pool, workers, size, kind, games, agents):
    res = {"size": size, "kind": kind, "games": games, "agents": {}}
    shortest = None
    for a in agents:
        jobs = [(a, size, kind, list(range(w, games, workers))) for w in range(workers)]
        parts = pool.map(_part, jobs) if pool else [_part(j) for j in jobs]
        rows = sorted((r for p in parts for r in p), key=lambda r: r[0])
        if any(r[2] is None for r in rows):
            continue
        shortest = [r[1] for r in rows]
        steps = [r[2][0] for r in rows]
        ok = [r[2][1] for r in rows]
        res["agents"][a] = {
            "mean_steps": round(sum(steps) / games, 2),
            "ratio": round(sum(s / max(sh, 1) for s, sh in zip(steps, shortest)) / games, 3),
            "success": round(sum(ok) / games, 4),
            "sample": [[r[0], r[2][0], r[2][1]] for r in rows[:SAMPLE]],
        }
    res["mean_shortest"] = round(sum(shortest) / games, 2) if shortest else None
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m game.train.passport")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from game.ai.net import MODEL_DIR, PASSPORT, STRONG, load_model
    from aikit.storage import load_json
    started = time.time()
    checks = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for size, kind in CHECKS:
            t0 = time.time()
            c = run_check(pool, args.workers, size, kind, args.games, AGENTS)
            checks.append(c)
            print(f"{size}x{size} {kind}: " + ", ".join(f"{a} {v['ratio']}x ({v['success']:.0%})"
                                                       for a, v in c["agents"].items())
                  + f" - {time.time() - t0:.0f} с", flush=True)
    net = load_model(STRONG)
    training = load_json(os.path.join(MODEL_DIR, "training.json")) or {}
    passport = {
        "model": STRONG,
        "network": {"input": "окно памяти 9x9 x 9 каналов + 8 общих (направление на выход, свой шаг)",
                    "hidden": list(net.hidden) if net else None, "parameters": net.count() if net else 0,
                    "play": "ход случайно по вероятностям сети, зерно - номер лабиринта"},
        "training": {k: training.get(k) for k in ("updates", "transitions", "episodes", "seconds", "threads",
                                                  "early_at")},
        "agents": [a for a in AGENTS if any(a in c["agents"] for c in checks)],
        "games_per_check": args.games,
        "seed": SEED,
        "checks": checks,
        "seconds": round(time.time() - started, 1),
    }
    out = args.out or os.path.join(MODEL_DIR, PASSPORT)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(passport, f, ensure_ascii=False, indent=1)
    print(f"паспорт -> {out}, {passport['seconds'] / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
