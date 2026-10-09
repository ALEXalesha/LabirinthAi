import numpy as np
import pytest

from aikit.nn import io, layers, losses
from aikit.nn.gradcheck import check_layer, numeric_grad
from aikit.nn.optim import Adam


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_conv3x3_gradients(rng):
    layer = layers.Conv3x3(3, 4, rng, dtype=np.float64)
    check_layer(layer, rng.normal(size=(2, 5, 4, 3)), rng)


def test_conv1x1_gradients(rng):
    layer = layers.Conv1x1(3, 4, rng, dtype=np.float64)
    check_layer(layer, rng.normal(size=(2, 3, 3, 3)), rng)


def test_dense_gradients(rng):
    layer = layers.Dense(5, 3, rng, dtype=np.float64)
    check_layer(layer, rng.normal(size=(4, 5)), rng)


def test_relu_gradients(rng):
    x = rng.normal(size=(3, 4))
    x[np.abs(x) < 0.05] = 0.5  # подальше от излома
    check_layer(layers.ReLU(), x, rng)


def test_global_max_gradients(rng):
    check_layer(layers.GlobalMax(), rng.normal(size=(2, 4, 3, 5)), rng)


def test_global_max_sends_gradient_only_to_the_maximum():
    x = np.zeros((1, 3, 3, 1))
    x[0, 2, 1, 0] = 5.0
    gm = layers.GlobalMax()
    assert gm.forward(x)[0, 0] == 5.0
    dx = gm.backward(np.array([[2.0]]))
    assert dx[0, 2, 1, 0] == 2.0
    assert dx.sum() == 2.0


def test_conv3x3_with_centre_kernel_equals_conv1x1(rng):
    c3 = layers.Conv3x3(3, 2, rng, dtype=np.float64)
    c1 = layers.Conv1x1(3, 2, rng, dtype=np.float64)
    c3.W[...] = 0
    c3.W[1, 1] = c1.W
    c3.b[...] = c1.b
    x = rng.normal(size=(2, 4, 5, 3))
    np.testing.assert_allclose(c3.forward(x), c1.forward(x))


def test_conv3x3_sees_zero_outside_the_field():
    c = layers.Conv3x3(1, 1, np.random.default_rng(0), dtype=np.float64)
    c.W[...] = 1.0
    c.b[...] = 0.0
    y = c.forward(np.ones((1, 3, 3, 1)))[0, :, :, 0]
    np.testing.assert_array_equal(y, [[4, 6, 4], [6, 9, 6], [4, 6, 4]])


def test_conv3x3_works_on_any_field_size(rng):
    c = layers.Conv3x3(2, 3, rng)
    for h, w in [(1, 1), (8, 13), (32, 32)]:
        assert c.forward(np.zeros((1, h, w, 2), np.float32)).shape == (1, h, w, 3)


def test_softmax_xent_gradient(rng):
    logits = rng.normal(size=(3, 4, 5))
    target = rng.integers(0, 5, size=(3, 4))
    weight = rng.uniform(0, 2, size=(3, 4))
    _, grad = losses.softmax_xent(logits, target, weight)
    num = numeric_grad(lambda: losses.softmax_xent(logits, target, weight)[0], logits)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-7)


def test_softmax_xent_ignores_zero_weight(rng):
    logits = rng.normal(size=(2, 3))
    loss, grad = losses.softmax_xent(logits, np.array([0, 1]), np.array([1.0, 0.0]))
    assert np.all(grad[1] == 0)
    p = np.exp(logits[0]) / np.exp(logits[0]).sum()
    assert loss == pytest.approx(-np.log(p[0]))


def test_bce_gradient(rng):
    logits = rng.normal(size=(4, 3))
    target = rng.integers(0, 2, size=(4, 3)).astype(float)
    _, grad = losses.bce(logits, target)
    num = numeric_grad(lambda: losses.bce(logits, target)[0], logits)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-7)


def test_bce_is_stable_on_large_logits():
    loss, grad = losses.bce(np.array([[1000.0, -1000.0]]), np.array([[1.0, 0.0]]))
    assert np.isfinite(loss) and loss < 1e-6
    assert np.all(np.isfinite(grad))


def test_adam_minimises_a_quadratic(rng):
    w = rng.normal(size=5)
    g = np.zeros_like(w)
    opt = Adam([("w", w, g)], lr=0.05)
    start = float((w ** 2).sum())
    for _ in range(300):
        g[...] = 2 * w
        opt.step()
    assert float((w ** 2).sum()) < start / 100


def test_save_and_load_round_trip(tmp_path, rng):
    arrays = {"a": rng.normal(size=(2, 3)).astype(np.float32), "b": np.arange(4)}
    path = tmp_path / "m.npz"
    io.save(path, arrays)
    back = io.load(path)
    assert set(back) == {"a", "b"}
    for k in arrays:
        np.testing.assert_array_equal(back[k], arrays[k])


# --- tanh, flatten, последовательность, mse ---


def test_tanh_gradients(rng):
    check_layer(layers.Tanh(), rng.normal(size=(3, 4)), rng)


def test_tanh_output_is_inside_minus_one_one(rng):
    y = layers.Tanh().forward(rng.normal(size=(50,)) * 100)
    assert np.all(np.abs(y) <= 1.0)


def test_flatten_gradients(rng):
    check_layer(layers.Flatten(), rng.normal(size=(2, 3, 3, 2)), rng)


def test_flatten_keeps_batch_axis(rng):
    assert layers.Flatten().forward(rng.normal(size=(4, 5, 5, 3))).shape == (4, 75)


def test_sequential_gradients(rng):
    net = layers.Sequential([
        layers.Conv3x3(2, 3, rng, dtype=np.float64), layers.ReLU(),
        layers.Flatten(), layers.Dense(3 * 3 * 3, 4, rng, dtype=np.float64), layers.Tanh(),
        layers.Dense(4, 1, rng, dtype=np.float64), layers.Tanh()])
    x = rng.normal(size=(2, 3, 3, 2))
    check_layer(net, x, rng)


def test_sequential_params_name_every_layer(rng):
    net = layers.Sequential([layers.Dense(2, 3, rng), layers.ReLU(), layers.Dense(3, 1, rng)])
    names = [name for name, _, _ in net.params()]
    assert names == ["0.W", "0.b", "2.W", "2.b"]


def test_mse_gradient(rng):
    pred = rng.normal(size=(5, 1))
    target = rng.normal(size=(5, 1))
    loss, grad = losses.mse(pred, target)
    num = numeric_grad(lambda: losses.mse(pred, target)[0], pred)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-7)
    assert loss == pytest.approx(float(((pred - target) ** 2).mean()))


def test_mse_weights(rng):
    pred = np.array([[1.0], [3.0]])
    loss, grad = losses.mse(pred, np.zeros((2, 1)), weight=np.array([[1.0], [0.0]]))
    assert loss == pytest.approx(1.0)
    assert grad[1, 0] == 0


def test_sequential_save_and_restore_gives_same_output(tmp_path, rng):
    def make(seed):
        r = np.random.default_rng(seed)
        return layers.Sequential([layers.Dense(4, 6, r), layers.Tanh(), layers.Dense(6, 1, r)])
    a, b = make(1), make(2)
    x = rng.normal(size=(3, 4)).astype(np.float32)
    assert not np.allclose(a.forward(x), b.forward(x))
    path = tmp_path / "net.npz"
    io.save(path, io.collect({"net": a}))
    io.restore({"net": b}, io.load(path))
    np.testing.assert_array_equal(a.forward(x), b.forward(x))


def test_restore_refuses_other_shapes(tmp_path, rng):
    a = layers.Sequential([layers.Dense(4, 6, rng)])
    b = layers.Sequential([layers.Dense(4, 5, rng)])
    with pytest.raises(ValueError):
        io.restore({"net": b}, io.collect({"net": a}))
