"""Обучение по команде:

    python -m game.train --trial                       # пробный прогон и прогноз времени
    python -m game.train --updates 600                 # с нуля
    python -m game.train --updates 900 --resume        # продолжить со снимка до 900 кругов
    python -m game.train --publish                     # сети из прогона -> game/models

Потоков BLAS --threads (по умолчанию 2): ПК делят и другие задачи. Журнал, снимок, ранняя
сеть (для среднего уровня) - в --run (runs/labirinth).
"""
import os
import sys


def _threads(argv):
    n = "2"
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith("--threads="):
            n = a.split("=", 1)[1]
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = n


_threads(sys.argv)

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m game.train")
    ap.add_argument("--run", default=os.path.join("runs", "labirinth"))
    ap.add_argument("--updates", type=int, default=600, help="кругов PPO всего")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--shaping", type=float, default=0.0, help="добавка за шаг к выходу по прямой")
    ap.add_argument("--new-cell", type=float, default=0.005, help="добавка за шаг в новую клетку")
    ap.add_argument("--sizes", default="5,20")
    ap.add_argument("--hints", action="store_true", help="сеть берёт подсказку-прикидку пути (teacher.hint)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--early-at", type=int, default=0, help="на каком круге сохранить раннюю сеть "
                    "(по умолчанию - 1/10 всех кругов)")
    ap.add_argument("--threads", default="2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="начать с весов сети (например, после подражания: runs/labirinth-imitate/net.npz)")
    ap.add_argument("--value-warmup", type=int, default=0, help="сколько первых кругов учить только оценку")
    ap.add_argument("--trial", action="store_true", help="3 круга и прогноз времени")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args(argv)

    from game.ai.net import EARLY, MODEL_DIR, STRONG, PolicyNet
    from game.train.ppo import Trainer
    hidden = tuple(int(x) for x in args.hidden.split(","))
    kw = dict(envs=args.envs, steps=args.steps, hidden=hidden, lr=args.lr, ent=args.ent, epochs=args.epochs,
              seed=args.seed, gamma=args.gamma, shaping=args.shaping, new_cell=args.new_cell,
              sizes=tuple(int(x) for x in args.sizes.split(",")), hints=args.hints)

    if args.publish:
        os.makedirs(MODEL_DIR, exist_ok=True)
        t = Trainer(args.run, echo=False, **kw)
        if not t.load_snapshot():
            print("нет снимка в", args.run)
            return 1
        t.net.save(os.path.join(MODEL_DIR, STRONG))
        early = os.path.join(args.run, "early.npz")
        if os.path.exists(early):
            shutil.copyfile(early, os.path.join(MODEL_DIR, EARLY))
        with open(os.path.join(MODEL_DIR, "training.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"cfg": t.cfg, "updates": t.update, "transitions": t.transitions,
                       "episodes": t.episodes, "seconds": round(t.seconds, 1), "threads": args.threads,
                       "early_at": _early_at(args), "curve_key": "ratio_sampled",
                       "baselines": baselines(t.valid), "curve": t.curve}, f, ensure_ascii=False, indent=1)
        print(f"сеть круга {t.update} -> game/models")
        return 0

    if args.trial:
        run = tempfile.mkdtemp(prefix="labirinth-trial-")
        t = Trainer(run, echo=False, **kw)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            t.iterate()
            times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        t.checkpoint_curve()
        eval_s = time.perf_counter() - t0
        per = min(times[1:])
        evals = args.updates // args.eval_every
        total = per * args.updates + eval_s * evals
        print(json.dumps({"sec_per_update": round(per, 2), "transitions_per_update": args.envs * args.steps,
                          "eval_s": round(eval_s, 1), "updates": args.updates, "threads": args.threads,
                          "forecast_minutes": round(total / 60, 1),
                          "note": "оценка до обучения медленнее: случайная сеть долго блуждает"},
                         ensure_ascii=False))
        return 0

    t = Trainer(args.run, **kw)
    if args.resume and t.load_snapshot():
        t.log(f"продолжаю с круга {t.update}")
    else:
        if args.init:
            t.net.copy_from(PolicyNet.load(args.init))
            t.log(f"веса из {args.init}")
        t.log(f"с нуля: {json.dumps(t.cfg, ensure_ascii=False)}, потоков {args.threads}, сеть {t.net.count()} весов")
        t.log(f"кривая: {t.checkpoint_curve()}")
    early_at = _early_at(args)
    while t.update < args.updates:
        info = t.iterate(value_only=t.update < args.value_warmup)
        if t.update % 5 == 0:
            t.log(f"круг {t.update}: обучение шаги/кратчайший {info['train_ratio']}, дошли {info['train_success']}, "
                  f"энтропия {info['entropy']:.3f}, отсечено {info['clipfrac']:.2f}, "
                  f"{info['collect_s'] + info['opt_s']:.1f} с")
        if t.update == early_at:
            t.net.save(os.path.join(args.run, "early.npz"))
            t.log(f"ранняя сеть сохранена (круг {t.update})")
        if t.update % args.eval_every == 0 or t.update == args.updates:
            t.log(f"кривая: {t.checkpoint_curve()}")
            t.save_snapshot()
    t.save_snapshot()
    t.log(f"готово: {t.update} кругов, {t.transitions} шагов, {t.episodes} лабиринтов, {t.seconds / 60:.1f} мин")
    return 0


def baselines(mazes):
    """Шаги / кратчайший у классических решателей на тех же лабиринтах, что кривая сети."""
    from game.ai.agents import make_agent
    from game.core.fog import play
    from game.train.ppo import score
    out = {}
    for name in ("right_hand", "tremaux", "optimist"):
        res = []
        for i, m in enumerate(mazes):
            run = play(make_agent(name, seed=i), m)
            res.append((run.memory.steps, run.done))
        out[name] = round(score(res, mazes)[0], 3)
    return out


def _early_at(args):
    return args.early_at or max(1, args.updates // 10)


if __name__ == "__main__":
    sys.exit(main())
