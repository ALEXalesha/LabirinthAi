"""Функции потерь: значение и градиент по логитам одним вызовом."""
import numpy as np


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_xent(logits, target, weight=None):
    """Перекрёстная энтропия по последней оси. target - номера классов формы logits[..., 0],
    weight - вес каждого примера той же формы (0 - пример не учится). Потеря - взвешенное
    среднее: делится на сумму весов."""
    if weight is None:
        weight = np.ones(target.shape)
    p = softmax(logits)
    picked = np.take_along_axis(p, target[..., None], axis=-1)[..., 0]
    total = max(float(weight.sum()), 1e-12)
    loss = float(-(weight * np.log(np.maximum(picked, 1e-30))).sum() / total)
    grad = p.copy()
    np.put_along_axis(grad, target[..., None], picked[..., None] - 1.0, axis=-1)
    grad *= (weight / total)[..., None]
    return loss, grad.astype(logits.dtype)


def bce(logits, target, weight=None):
    """Двоичная перекрёстная энтропия по сигмоиде логитов, среднее по весам.
    Считается через log(1 + e^-|x|) - без переполнения на больших логитах."""
    if weight is None:
        weight = np.ones(target.shape)
    total = max(float(weight.sum()), 1e-12)
    per = np.maximum(logits, 0) - logits * target + np.log1p(np.exp(-np.abs(logits)))
    loss = float((weight * per).sum() / total)
    sig = 0.5 * (1.0 + np.tanh(0.5 * logits))
    grad = (sig - target) * weight / total
    return loss, grad.astype(logits.dtype)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def mse(pred, target, weight=None):
    """Средний квадрат ошибки, взвешенное среднее: делится на сумму весов."""
    if weight is None:
        weight = np.ones(pred.shape)
    total = max(float(weight.sum()), 1e-12)
    diff = pred - target
    loss = float((weight * diff * diff).sum() / total)
    grad = 2.0 * weight * diff / total
    return loss, grad.astype(pred.dtype)
