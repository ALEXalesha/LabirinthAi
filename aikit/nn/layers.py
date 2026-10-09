"""Слои сети на numpy. Раскладка тензоров (B, H, W, C) - канал последним, так свёртка
сводится к одному умножению матриц.

У каждого слоя forward(x) -> y, backward(dy) -> dx (градиенты весов копятся в self.d*),
params() -> [(имя, массив, градиент)] - по этому списку работает Adam и сохранение.
"""
import numpy as np


_WINDOWS = {}


def _window_index(H, W):
    """Номера клеток поля с паддингом (H+2)x(W+2), из которых собирается окно 3x3 каждой
    клетки: (H*W*9,). Раньше окна резались sliding_window_view с np.pad - в поиске, где
    поле 8x8 и пачки маленькие, это занимало две трети времени."""
    key = (H, W)
    idx = _WINDOWS.get(key)
    if idx is None:
        y, x, i, j = np.meshgrid(np.arange(H), np.arange(W), np.arange(3), np.arange(3), indexing="ij")
        idx = ((y + i) * (W + 2) + (x + j)).reshape(-1)
        _WINDOWS[key] = idx
    return idx


def _he(rng, fan_in, shape, dtype):
    return (rng.normal(size=shape) * np.sqrt(2.0 / fan_in)).astype(dtype)


class Conv3x3:
    """Свёртка 3x3 с паддингом нулями: выход того же размера, за краем поля - нули.
    Размер поля не зашит - одна и та же сеть работает на 8x8 и на 32x32."""

    def __init__(self, cin, cout, rng, dtype=np.float32):
        self.W = _he(rng, 9 * cin, (3, 3, cin, cout), dtype)
        self.b = np.zeros(cout, dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        B, H, W, C = x.shape
        xp = np.zeros((B, H + 2, W + 2, C), x.dtype)
        xp[:, 1:-1, 1:-1] = x
        # окно 3x3 каждой клетки одним индексом по плоскому полю с паддингом; порядок
        # (клетка, i, j, канал) - как у весов (3, 3, C, cout)
        self.cols = xp.reshape(B, (H + 2) * (W + 2), C)[:, _window_index(H, W)].reshape(B * H * W, 9 * C)
        self.shape = x.shape
        y = self.cols @ self.W.reshape(9 * C, -1) + self.b
        return y.reshape(B, H, W, -1)

    def backward(self, dy):
        B, H, W, C = self.shape
        d2 = dy.reshape(B * H * W, -1)
        self.dW[...] = (self.cols.T @ d2).reshape(self.dW.shape)
        self.db[...] = d2.sum(axis=0)
        dcols = (d2 @ self.W.reshape(9 * C, -1).T).reshape(B, H, W, 3, 3, C)
        dxp = np.zeros((B, H + 2, W + 2, C), dy.dtype)
        for i in range(3):
            for j in range(3):
                dxp[:, i:i + H, j:j + W] += dcols[:, :, :, i, j]
        return dxp[:, 1:-1, 1:-1]

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class Dense:
    """Плотный слой по последней оси: работает и на (B, n), и на (B, H, W, n) -
    на поле это свёртка 1x1."""

    def __init__(self, nin, nout, rng, dtype=np.float32):
        self.W = _he(rng, nin, (nin, nout), dtype)
        self.b = np.zeros(nout, dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dy):
        n = self.W.shape[0]
        self.dW[...] = self.x.reshape(-1, n).T @ dy.reshape(-1, dy.shape[-1])
        self.db[...] = dy.reshape(-1, dy.shape[-1]).sum(axis=0)
        return dy @ self.W.T

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


Conv1x1 = Dense


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask

    def params(self):
        return []


class GlobalMax:
    """Максимум по всему полю в каждом канале: (B, H, W, C) -> (B, C). Так сеть узнаёт
    о событии, которое случилось в одной клетке где угодно на поле."""

    def forward(self, x):
        B, H, W, C = x.shape
        flat = x.reshape(B, H * W, C)
        self.idx = flat.argmax(axis=1)
        self.shape = x.shape
        return np.take_along_axis(flat, self.idx[:, None, :], axis=1)[:, 0]

    def backward(self, dy):
        B, H, W, C = self.shape
        dx = np.zeros((B, H * W, C), dy.dtype)
        np.put_along_axis(dx, self.idx[:, None, :], dy[:, None, :], axis=1)
        return dx.reshape(self.shape)

    def params(self):
        return []


class Tanh:
    """Гиперболический тангенс: выход сети оценки в (-1, 1) - от проигрыша до победы."""

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dy):
        return dy * (1.0 - self.y * self.y)

    def params(self):
        return []


class Flatten:
    """(B, ...) -> (B, n): поле после свёрток в вектор для плотного выхода."""

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(self.shape)

    def params(self):
        return []


class Sequential:
    """Слои по порядку. Имена весов - номер слоя и имя в нём: '0.W', '2.b'."""

    def __init__(self, items):
        self.items = list(items)

    def forward(self, x):
        for layer in self.items:
            x = layer.forward(x)
        return x

    def backward(self, dy):
        for layer in reversed(self.items):
            dy = layer.backward(dy)
        return dy

    def params(self):
        return [(f"{i}.{name}", p, g) for i, layer in enumerate(self.items)
                for name, p, g in layer.params()]
