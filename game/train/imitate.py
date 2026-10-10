"""Подражание Оптимисту (DAgger), потом - дообучение PPO (python -m game.train --init ...).

Решение Алексея 10.10.2026: чистый PPO не догнал Оптимиста (4.57x кратчайшего против
1.70x), поэтому сеть сначала учится повторять за ним, а потом доучивается сама.

Оптимист честный - видит только память игрока (game/ai/teacher.py). Сеть по-прежнему
получает только features.encode. DAgger: лабиринты проходит смесь - с вероятностью beta
ходит учитель, иначе сама сеть (beta падает от 1 до 0), а учитель подписывает каждое
состояние своим ходом. Так сеть видит и свои ошибки, и как из них выбираться.

    python -m game.train.imitate --iters 300                  # с нуля
    python -m game.train.imitate --iters 400 --resume
    python -m game.train.imitate --trial                      # прогноз времени
"""
import os
import sys


def _threads(argv):
    n = "2"
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = n


if __name__ == "__main__":
    _threads(sys.argv)

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from aikit.nn.losses import softmax, softmax_xent  # noqa: E402
from aikit.nn.optim import Adam  # noqa: E402
from aikit.trainlog import TrainLog, load_snapshot, save_snapshot  # noqa: E402
from game.ai import features as F  # noqa: E402
from game.ai import teacher  # noqa: E402
from game.ai.env import BatchEnv, net_chooser, random_maze, run_policy  # noqa: E402
from game.ai.net import PolicyNet  # noqa: E402
from game.train.ppo import score, validation_mazes  # noqa: E402


class Imitator:
    def __init__(self, run_dir, envs=64, steps=64, hidden=(128, 128), lr=1e-3, seed=1, sizes=(5, 20),
                 cap_mult=4, buffer=100_000, batch=1024, updates_per_iter=16, beta_iters=100, valid_per=25,
                 hints=False, echo=True):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log = TrainLog(os.path.join(run_dir, "train.log"), echo=echo)
        self.snapshot_path = os.path.join(run_dir, "snapshot.npz")
        self.cfg = dict(envs=envs, steps=steps, hidden=list(hidden), lr=lr, seed=seed, sizes=list(sizes),
                        cap_mult=cap_mult, buffer=buffer, batch=batch, updates_per_iter=updates_per_iter,
                        beta_iters=beta_iters, hints=hints)
        self.hints = hints
        self.net = PolicyNet(hidden, seed=seed, nin=F.SIZE_HINT if hints else F.SIZE)
        self.opt = Adam(self.net.params(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.env = BatchEnv(envs, sizes[1], sizes[1])
        for i in range(envs):
            self._new_maze(i)
        self.xs = np.zeros((buffer, self.net.nin), np.float16)
        self.legal = np.zeros((buffer, 4), bool)
        self.label = np.zeros(buffer, np.int64)
        self.filled = 0
        self.head = 0
        self.it = 0
        self.transitions = 0
        self.episodes = 0
        self.seconds = 0.0
        self.curve = []
        self.valid = validation_mazes(valid_per)

    def _new_maze(self, i):
        m = random_maze(self.rng, tuple(self.cfg["sizes"]))
        self.env.load(i, m, self.cfg["cap_mult"] * m.h * m.w)

    def beta(self):
        return max(0.0, 1.0 - self.it / max(self.cfg["beta_iters"], 1))

    def _store(self, x, legal, label):
        n = len(label)
        idx = (self.head + np.arange(n)) % len(self.label)
        self.xs[idx] = x
        self.legal[idx] = legal
        self.label[idx] = label
        self.head = (self.head + n) % len(self.label)
        self.filled = min(self.filled + n, len(self.label))

    def collect(self):
        env, B = self.env, self.env.n
        beta = self.beta()
        agree = 0
        for _ in range(self.cfg["steps"]):
            x, flips, legal = env.encode(hints=self.hints)
            real_t = teacher.choose(env.known, env.mem, env.visits, env.pos, env.exit, env.hw)
            label = F.from_real(real_t, flips)
            logits, _ = self.net.forward(x, legal)
            p = softmax(logits.astype(np.float64))
            u = self.rng.random(B)[:, None]
            a = np.minimum((p.cumsum(axis=1) < u).sum(axis=1), 3)
            bad = ~legal[np.arange(B), a]
            a[bad] = logits[bad].argmax(axis=1)
            agree += int((logits.argmax(axis=1) == label).sum())
            use_teacher = self.rng.random(B) < beta
            act = np.where(use_teacher, real_t, F.to_real(a, flips))
            self._store(x, legal, label)
            _, done, trunc = env.step(act)
            for i in np.nonzero(done | trunc)[0]:
                self.episodes += 1
                self._new_maze(i)
        self.transitions += self.cfg["steps"] * B
        return agree / (self.cfg["steps"] * B)

    def train(self):
        n = self.filled
        losses = []
        for _ in range(self.cfg["updates_per_iter"]):
            mb = self.rng.integers(0, n, self.cfg["batch"])
            x = self.xs[mb].astype(np.float32)
            logits, v = self.net.forward(x, self.legal[mb])
            loss, grad = softmax_xent(logits.astype(np.float64), self.label[mb])
            self.net.backward(grad.astype(np.float32), np.zeros_like(v))
            self.opt.step()
            losses.append(loss)
        return float(np.mean(losses))

    def iterate(self):
        t0 = time.perf_counter()
        agree = self.collect()
        t1 = time.perf_counter()
        loss = self.train()
        t2 = time.perf_counter()
        self.it += 1
        self.seconds += t2 - t0
        return {"loss": round(loss, 4), "agree": round(agree, 3), "beta": round(self.beta(), 2),
                "collect_s": round(t1 - t0, 2), "train_s": round(t2 - t1, 2)}

    def evaluate(self, greedy):
        res = run_policy(net_chooser(self.net, greedy, np.random.default_rng(0)), self.valid)
        return score(res, self.valid)

    def checkpoint_curve(self):
        r, s = self.evaluate(True)
        rs, ss = self.evaluate(False)
        point = {"iter": self.it, "transitions": self.transitions, "episodes": self.episodes,
                 "seconds": round(self.seconds, 1), "ratio": round(r, 3), "success": round(s, 3),
                 "ratio_sampled": round(rs, 3), "success_sampled": round(ss, 3)}
        self.curve.append(point)
        return point

    def save_snapshot(self):
        arrays = {name: p for name, p, _ in self.net.params()}
        for i, (m, v) in enumerate(zip(self.opt.m, self.opt.v)):
            arrays[f"adam.m.{i}"] = m
            arrays[f"adam.v.{i}"] = v
        meta = {"cfg": self.cfg, "it": self.it, "transitions": self.transitions, "episodes": self.episodes,
                "seconds": self.seconds, "curve": self.curve, "adam_t": self.opt.t,
                "rng": self.rng.bit_generator.state}
        save_snapshot(self.snapshot_path, arrays, meta)
        self.net.save(os.path.join(self.run_dir, "net.npz"))

    def load_snapshot(self):
        got = load_snapshot(self.snapshot_path)
        if got is None:
            return False
        arrays, meta = got
        for name, p, _ in self.net.params():
            p[...] = arrays[name]
        for i, (m, v) in enumerate(zip(self.opt.m, self.opt.v)):
            m[...] = arrays[f"adam.m.{i}"]
            v[...] = arrays[f"adam.v.{i}"]
        self.opt.t = meta["adam_t"]
        self.it, self.transitions, self.episodes = meta["it"], meta["transitions"], meta["episodes"]
        self.seconds, self.curve = meta["seconds"], meta["curve"]
        self.rng.bit_generator.state = meta["rng"]
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m game.train.imitate")
    ap.add_argument("--run", default=os.path.join("runs", "labirinth-imitate"))
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--beta-iters", type=int, default=100, help="за сколько кругов учитель уступает сети")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--threads", default="2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--hints", action="store_true", help="сеть берёт подсказку-прикидку пути (teacher.hint)")
    args = ap.parse_args(argv)
    if args.trial:
        t = Imitator(tempfile.mkdtemp(prefix="labirinth-imitate-trial-"), echo=False, beta_iters=args.beta_iters,
                     hints=args.hints)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            t.iterate()
            times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        t.checkpoint_curve()
        ev = time.perf_counter() - t0
        per = min(times[1:])
        total = per * args.iters + ev * (args.iters // args.eval_every)
        print(json.dumps({"sec_per_iter": round(per, 2), "eval_s": round(ev, 1), "iters": args.iters,
                          "threads": args.threads, "forecast_minutes": round(total / 60, 1)}, ensure_ascii=False))
        return 0
    t = Imitator(args.run, beta_iters=args.beta_iters, hints=args.hints)
    if args.resume and t.load_snapshot():
        t.log(f"продолжаю с круга {t.it}")
    else:
        t.log(f"подражание с нуля: {json.dumps(t.cfg, ensure_ascii=False)}, потоков {args.threads}")
    while t.it < args.iters:
        info = t.iterate()
        if t.it % 5 == 0:
            t.log(f"круг {t.it}: потеря {info['loss']}, совпадает с учителем {info['agree']}, beta {info['beta']}, "
                  f"{info['collect_s'] + info['train_s']:.1f} с")
        if t.it % args.eval_every == 0 or t.it == args.iters:
            t.log(f"кривая: {t.checkpoint_curve()}")
            t.save_snapshot()
    t.save_snapshot()
    t.log(f"готово: {t.it} кругов, {t.transitions} шагов, {t.episodes} лабиринтов, {t.seconds / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    sys.exit(main())
