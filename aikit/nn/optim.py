import numpy as np


class Adam:
    """Adam по списку (имя, массив, градиент); массивы меняются на месте."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for _, p, _ in self.params]
        self.v = [np.zeros_like(p) for _, p, _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for (_, p, g), m, v in zip(self.params, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= (self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)
