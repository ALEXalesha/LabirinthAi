"""Обучение с подкреплением (PPO) на тысячах разных лабиринтов.

Каждый шаг стоит -0.01, шаг ближе к выходу (по прямой) - на 0.005 дешевле, шаг дальше -
на 0.005 дороже. Такая добавка - разность потенциала (расстояния до выхода): лучший путь
от неё не меняется, а учиться в начале легче. Лабиринт пройден - новый, случайный:
генератор DFS, Прим, Краскал или Уилсон, размер от 5 до 20 по каждой стороне, иногда с
петлями, вход и выход в случайной паре противоположных углов. Двоичное дерево и комнаты
сеть при обучении не видит - на них проверяется, как она справляется с незнакомым.

Сеть видит только память игрока (features.encode). Журнал и снимок - в папке прогона.
"""
import os
import time

import numpy as np

from aikit.nn.losses import softmax
from aikit.nn.optim import Adam
from aikit.trainlog import TrainLog, load_snapshot, save_snapshot
from game.ai import features as F
from game.ai.env import TRAIN_KINDS, BatchEnv, net_chooser, random_maze, run_policy
from game.ai.net import PolicyNet
from game.core.generators import generate

ENV_FIELDS = ("true", "known", "mem", "visits", "pos", "exit", "hw", "last", "steps", "cap", "shortest", "active")
VALID_SEED = 3_000_000_000          # зёрна проверок - вне диапазона обучения (< 2^30)


def ppo_loss(logits, value, act, old_logp, adv, ret, clip=0.2, ent=0.01, vf=0.5):
    """Потеря PPO по пачке и её градиент по логитам и оценке.
    logits - с -1e9 у запрещённых ходов. Возвращает (потеря, dlogits, dvalue, сведения)."""
    M = len(act)
    p = softmax(logits.astype(np.float64))
    logp_all = np.log(np.maximum(p, 1e-30))
    rows = np.arange(M)
    logp = logp_all[rows, act]
    ratio = np.exp(logp - old_logp)
    clipped = np.clip(ratio, 1 - clip, 1 + clip)
    pg = -np.minimum(ratio * adv, clipped * adv).mean()
    H = -(p * logp_all).sum(axis=1)
    v_loss = ((value - ret) ** 2).mean()
    loss = pg + vf * v_loss - ent * H.mean()

    active = ~(((adv >= 0) & (ratio > 1 + clip)) | ((adv < 0) & (ratio < 1 - clip)))
    g = np.where(active, -adv * ratio, 0.0) / M
    onehot = np.zeros_like(p)
    onehot[rows, act] = 1.0
    dlogits = g[:, None] * (onehot - p)
    dlogits += ent / M * p * (logp_all + H[:, None])
    dvalue = vf * 2.0 * (value - ret) / M
    info = {"pg": float(pg), "v": float(v_loss), "entropy": float(H.mean()),
            "clipfrac": float((np.abs(ratio - 1) > clip).mean())}
    return float(loss), dlogits.astype(np.float32), dvalue.astype(np.float32), info


def gae(rewards, values, ends, last_value, gamma, lam):
    """Преимущества и цели оценки. ends[t] - после шага t эпизод кончился (дошёл или
    упёрся в предел; во втором случае оценка хвоста уже в награде)."""
    T = len(rewards)
    adv = np.zeros_like(rewards)
    nxt_adv = 0.0
    nxt_v = last_value
    for t in range(T - 1, -1, -1):
        keep = 1.0 - ends[t]
        delta = rewards[t] + gamma * nxt_v * keep - values[t]
        nxt_adv = delta + gamma * lam * keep * nxt_adv
        adv[t] = nxt_adv
        nxt_v = values[t]
    return adv, adv + values


def validation_mazes(per=25, sizes=(8, 12, 16, 20), kinds=TRAIN_KINDS):
    """Незнакомые лабиринты для кривой обучения: одни и те же на всех кругах."""
    out = []
    for n in sizes:
        for i in range(per):
            kind = kinds[i % len(kinds)]
            m = generate(kind, n, n, VALID_SEED + 1000 * n + i, loops=0.25 if i % 3 == 0 else 0.0)
            out.append(m.transformed(bool(i & 4), bool(i & 8)))
    return out


def score(results, mazes):
    """(средние шаги / кратчайший путь, доля дошедших)."""
    ratios = [s / max(m.shortest(), 1) for (s, _), m in zip(results, mazes)]
    return float(np.mean(ratios)), float(np.mean([ok for _, ok in results]))


class Trainer:
    def __init__(self, run_dir, envs=64, steps=128, hidden=(128, 128), lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=4, minibatch=2048, ent=0.01, vf=0.5, seed=1, sizes=(5, 20),
                 cap_mult=4, valid_per=25, shaping=0.0, new_cell=0.005, echo=True):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log = TrainLog(os.path.join(run_dir, "train.log"), echo=echo)
        self.snapshot_path = os.path.join(run_dir, "snapshot.npz")
        self.cfg = dict(envs=envs, steps=steps, hidden=list(hidden), lr=lr, gamma=gamma, lam=lam, clip=clip,
                        epochs=epochs, minibatch=minibatch, ent=ent, vf=vf, seed=seed, sizes=list(sizes),
                        cap_mult=cap_mult, shaping=shaping, new_cell=new_cell)
        self.net = PolicyNet(hidden, seed=seed)
        self.opt = Adam(self.net.params(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.env = BatchEnv(envs, sizes[1], sizes[1], shaping, new_cell)
        for i in range(envs):
            self._new_maze(i)
        self.update = 0
        self.transitions = 0
        self.episodes = 0
        self.seconds = 0.0
        self.curve = []
        self.recent = []
        self.valid = validation_mazes(valid_per)

    def _new_maze(self, i):
        m = random_maze(self.rng, tuple(self.cfg["sizes"]))
        self.env.load(i, m, self.cfg["cap_mult"] * m.h * m.w)

    def collect(self):
        c, env, net = self.cfg, self.env, self.net
        T, B = c["steps"], env.n
        xs = np.zeros((T, B, F.SIZE), np.float32)
        legals = np.zeros((T, B, 4), bool)
        acts = np.zeros((T, B), np.int64)
        logps = np.zeros((T, B))
        values = np.zeros((T, B))
        rewards = np.zeros((T, B))
        ends = np.zeros((T, B))
        for t in range(T):
            x, flips, legal = env.encode()
            logits, v = net.forward(x, legal)
            p = softmax(logits.astype(np.float64))
            u = self.rng.random(B)[:, None]
            a = np.minimum((p.cumsum(axis=1) < u).sum(axis=1), 3)
            bad = ~legal[np.arange(B), a]
            a[bad] = logits[bad].argmax(axis=1)
            r, done, trunc = env.step(F.to_real(a, flips))
            r = r.astype(np.float64)
            if trunc.any():
                ti = np.nonzero(trunc)[0]
                xt, _, lt = env.encode(ti)
                _, vt = net.forward(xt, lt)
                r[ti] += c["gamma"] * vt
            xs[t], legals[t], acts[t] = x, legal, a
            logps[t] = np.log(np.maximum(p[np.arange(B), a], 1e-30))
            values[t], rewards[t] = v, r
            ends[t] = done | trunc
            for i in np.nonzero(done | trunc)[0]:
                self.recent.append((env.steps[i] / max(env.shortest[i], 1), bool(done[i])))
                self.episodes += 1
                self._new_maze(i)
        x, _, legal = env.encode()
        _, last_v = net.forward(x, legal)
        adv, ret = gae(rewards, values, ends, last_v.astype(np.float64), c["gamma"], c["lam"])
        self.transitions += T * B
        return (xs.reshape(T * B, -1), legals.reshape(T * B, 4), acts.reshape(-1), logps.reshape(-1),
                adv.reshape(-1), ret.reshape(-1))

    def optimize(self, batch, value_only=False):
        c = self.cfg
        xs, legals, acts, logps, adv, ret = batch
        n = len(acts)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        info = {}
        for _ in range(c["epochs"]):
            perm = self.rng.permutation(n)
            for s in range(0, n, c["minibatch"]):
                mb = perm[s:s + c["minibatch"]]
                logits, v = self.net.forward(xs[mb], legals[mb])
                _, dl, dv, info = ppo_loss(logits, v.astype(np.float64), acts[mb], logps[mb], adv[mb], ret[mb],
                                           c["clip"], c["ent"], c["vf"])
                if value_only:
                    dl[...] = 0          # разминка оценки: ходы сети (например, после подражания) не трогаются
                self.net.backward(dl, dv)
                grads = [g for _, _, g in self.net.params()]
                norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads)))
                if norm > 0.5:
                    for g in grads:
                        g *= 0.5 / norm
                self.opt.step()
        return info

    def iterate(self, value_only=False):
        t0 = time.perf_counter()
        batch = self.collect()
        t1 = time.perf_counter()
        info = self.optimize(batch, value_only)
        t2 = time.perf_counter()
        self.update += 1
        self.seconds += t2 - t0
        recent = self.recent[-400:]
        info.update(collect_s=round(t1 - t0, 3), opt_s=round(t2 - t1, 3),
                    train_ratio=round(float(np.mean([r for r, _ in recent])), 3) if recent else None,
                    train_success=round(float(np.mean([ok for _, ok in recent])), 3) if recent else None)
        return info

    def evaluate(self, mazes=None, greedy=True):
        mazes = mazes or self.valid
        res = run_policy(net_chooser(self.net, greedy, np.random.default_rng(0)), mazes)
        return score(res, mazes)

    def checkpoint_curve(self):
        ratio, success = self.evaluate(greedy=True)
        sratio, ssuccess = self.evaluate(greedy=False)
        point = {"update": self.update, "transitions": self.transitions, "episodes": self.episodes,
                 "seconds": round(self.seconds, 1), "ratio": round(ratio, 3), "success": round(success, 3),
                 "ratio_sampled": round(sratio, 3), "success_sampled": round(ssuccess, 3)}
        self.curve.append(point)
        return point

    def save_snapshot(self, path=None):
        arrays = {name: p for name, p, _ in self.net.params()}
        for i, (m, v) in enumerate(zip(self.opt.m, self.opt.v)):
            arrays[f"adam.m.{i}"] = m
            arrays[f"adam.v.{i}"] = v
        for k in ENV_FIELDS:
            arrays[f"env.{k}"] = getattr(self.env, k)
        meta = {"cfg": self.cfg, "update": self.update, "transitions": self.transitions,
                "episodes": self.episodes, "seconds": self.seconds, "curve": self.curve,
                "adam_t": self.opt.t, "rng": self.rng.bit_generator.state}
        save_snapshot(path or self.snapshot_path, arrays, meta)

    def load_snapshot(self, path=None):
        got = load_snapshot(path or self.snapshot_path)
        if got is None:
            return False
        arrays, meta = got
        for name, p, _ in self.net.params():
            p[...] = arrays[name]
        for i, (m, v) in enumerate(zip(self.opt.m, self.opt.v)):
            m[...] = arrays[f"adam.m.{i}"]
            v[...] = arrays[f"adam.v.{i}"]
        for k in ENV_FIELDS:
            if f"env.{k}" in arrays and arrays[f"env.{k}"].shape == getattr(self.env, k).shape:
                getattr(self.env, k)[...] = arrays[f"env.{k}"]
        self.opt.t = meta["adam_t"]
        self.update, self.transitions = meta["update"], meta["transitions"]
        self.episodes, self.seconds, self.curve = meta["episodes"], meta["seconds"], meta["curve"]
        self.rng.bit_generator.state = meta["rng"]
        return True
