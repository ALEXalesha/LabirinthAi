"""Сеть и обучение: градиенты потери PPO, сохранение, снимок, пара кругов обучения."""
import numpy as np
import pytest

from aikit.nn.gradcheck import numeric_grad
from game.ai import features as F
from game.ai.net import NEG, PolicyNet
from game.train.ppo import Trainer, gae, ppo_loss


def test_ppo_loss_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    M = 12
    legal = rng.random((M, 4)) < 0.7
    legal[np.arange(M), rng.integers(0, 4, M)] = True
    raw = rng.normal(size=(M, 4))
    act = np.array([rng.choice(np.nonzero(l)[0]) for l in legal])
    old = np.log(rng.uniform(0.2, 0.9, M))
    adv = rng.normal(size=M)
    ret = rng.normal(size=M)
    value = rng.normal(size=M)

    def masked():
        return np.where(legal, raw, NEG)

    def loss():
        return ppo_loss(masked(), value, act, old, adv, ret, clip=0.2, ent=0.05, vf=0.5)[0]

    _, dl, dv, _ = ppo_loss(masked(), value, act, old, adv, ret, clip=0.2, ent=0.05, vf=0.5)
    num_l = numeric_grad(loss, raw, eps=1e-5)
    np.testing.assert_allclose(np.where(legal, dl, 0), num_l, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(dv, numeric_grad(loss, value, eps=1e-5), rtol=1e-3, atol=1e-5)


def test_ppo_clipping_stops_gradient():
    logits = np.zeros((2, 4))
    act = np.array([0, 0])
    old = np.log(np.array([0.1, 0.9]))           # отношение 2.5 и 0.28 - оба за краем
    adv = np.array([1.0, -1.0])
    _, dl, _, _ = ppo_loss(logits, np.zeros(2), act, old, adv, np.zeros(2), ent=0.0)
    assert np.allclose(dl, 0)


def test_gae_stops_at_episode_end():
    r = np.array([[1.0], [1.0], [1.0]])
    v = np.zeros((3, 1))
    ends = np.array([[0.0], [1.0], [0.0]])
    adv, ret = gae(r, v, ends, np.array([10.0]), gamma=0.5, lam=1.0)
    assert ret[1, 0] == 1.0                          # конец - хвост не тянется
    assert ret[0, 0] == 1.0 + 0.5 * 1.0
    assert ret[2, 0] == 1.0 + 0.5 * 10.0


def test_net_backward_matches_numeric():
    net = PolicyNet((8,), seed=1, nin=6)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 6)).astype(np.float32)
    legal = np.ones((3, 4), bool)
    cl, cv = rng.normal(size=(3, 4)), rng.normal(size=3)

    def loss():
        lg, v = net.forward(x, legal)
        return float((lg * cl).sum() + (v * cv).sum())

    loss()
    net.backward(cl.astype(np.float32), cv.astype(np.float32))
    for name, p, g in net.params():
        num = numeric_grad(lambda: loss(), p, eps=1e-2)
        np.testing.assert_allclose(g, num, rtol=2e-2, atol=2e-3, err_msg=name)


def test_net_save_load_same_output(tmp_path):
    net = PolicyNet((16, 8), seed=4)
    rng = np.random.default_rng(2)
    x = rng.random((5, F.SIZE)).astype(np.float32)
    legal = np.ones((5, 4), bool)
    path = str(tmp_path / "n.npz")
    net.save(path)
    other = PolicyNet.load(path)
    a, b = net.forward(x, legal), other.forward(x, legal)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert other.hidden == (16, 8)


def test_forbidden_moves_get_zero_probability():
    net = PolicyNet(seed=5)
    x = np.random.default_rng(3).random((4, F.SIZE)).astype(np.float32)
    legal = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]], bool)
    p, _ = net.probs(x, legal)
    assert np.all(p[~legal] == 0) and np.allclose(p.sum(axis=1), 1)


@pytest.mark.slow
def test_two_rounds_snapshot_and_resume(tmp_path):
    kw = dict(envs=8, steps=16, hidden=(16,), minibatch=64, valid_per=2, echo=False)
    t = Trainer(str(tmp_path / "a"), **kw)
    t.iterate()
    t.iterate()
    t.checkpoint_curve()
    t.save_snapshot()
    u = Trainer(str(tmp_path / "a"), **kw)
    assert u.load_snapshot()
    assert u.update == 2 and u.transitions == 2 * 8 * 16 and len(u.curve) == 1
    for (_, a, _), (_, b, _) in zip(t.net.params(), u.net.params()):
        assert np.array_equal(a, b)
    # продолжение идёт так же, как без перерыва
    t.iterate()
    u.iterate()
    for (_, a, _), (_, b, _) in zip(t.net.params(), u.net.params()):
        assert np.allclose(a, b)
