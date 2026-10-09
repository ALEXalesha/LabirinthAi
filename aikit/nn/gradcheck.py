"""Проверка градиентов: backward слоя против центральной разности.

Каждый слой aikit.nn проходит её в тестах. Новый слой без такой проверки в набор не входит.
"""
import numpy as np


def numeric_grad(f, x, eps=1e-3):
    """Центральная разность по каждому элементу x (x меняется на месте и возвращается)."""
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        up = f()
        x[i] = old - eps
        down = f()
        x[i] = old
        g[i] = (up - down) / (2 * eps)
    return g


def check_layer(layer, x, rng, rtol=1e-4, atol=1e-6):
    """backward слоя против численного градиента по входу и по всем весам. Потеря - сумма
    выхода со случайными коэффициентами, чтобы градиенты были разными. Ошибка - AssertionError."""
    y = layer.forward(x)
    coef = rng.normal(size=y.shape)

    def loss():
        return float((layer.forward(x) * coef).sum())

    layer.forward(x)
    dx = layer.backward(coef)
    assert dx.shape == x.shape
    np.testing.assert_allclose(dx, numeric_grad(loss, x), rtol=rtol, atol=atol)
    layer.forward(x)
    layer.backward(coef)
    for name, p, grad in layer.params():
        np.testing.assert_allclose(grad, numeric_grad(loss, p), rtol=rtol, atol=atol, err_msg=name)
