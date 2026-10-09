"""Что видит сеть: окно памяти вокруг себя и направление на выход.

Всё считается пачкой (B лабиринтов сразу) - так одна и та же функция служит и обучению,
и игре (там B = 1). На вход идёт только память игрока (known, cells, visits) и открытые
всем сведения: размер поля, где выход, свой последний шаг.

Окно разворачивается так, чтобы выход всегда был снизу справа (dy >= 0, dx >= 0): сеть
учит одно правило вместо четырёх. Направления сети - в развёрнутом окне; to_real и
from_canon переводят их обратно.

Каналы каждой клетки окна (C = 9): внутри поля; видел; проходы N, E, S, W (у виденных
клеток); был здесь; сколько раз был (до 4, деленное на 4); выход.
"""
import numpy as np

K = 4                       # окно (2K+1) x (2K+1) клеток вокруг игрока
SIDE = 2 * K + 1
C = 9
GLOBAL = 8
SIZE = SIDE * SIDE * C + GLOBAL

# перестановка направлений N E S W при отражениях: по вертикали N<->S, по горизонтали E<->W
_FLIP_V = np.array([2, 1, 0, 3])
_FLIP_H = np.array([0, 3, 2, 1])


def planes(known, cells, visits, hw, exit_):
    """Плоскости каналов (B, H, W, C) по всему полю - медленный образец: тест сверяет с ним
    быстрое encode."""
    B, H, W = known.shape
    out = np.zeros((B, H, W, C), np.float32)
    rows = np.arange(H)[None, :, None]
    cols = np.arange(W)[None, None, :]
    inside = (rows < hw[:, 0, None, None]) & (cols < hw[:, 1, None, None])
    out[..., 0] = inside
    out[..., 1] = known
    seen = np.where(known, cells, 0)
    for d, bit in enumerate((1, 2, 4, 8)):
        out[..., 2 + d] = (seen & bit) > 0
    out[..., 6] = visits > 0
    out[..., 7] = np.minimum(visits, 4) / 4.0
    out[np.arange(B), exit_[:, 0], exit_[:, 1], 8] = 1.0
    return out


def pad(a):
    """Поля шириной K вокруг поля (B, H, W): окно у края не выходит за массив."""
    return np.pad(a, ((0, 0), (K, K), (K, K)))


def encode(known, cells, visits, pos, exit_, hw, last, padded=None, rows=None):
    """Вход сети (B, SIZE), отражения flips (B, 2) и разрешённые ходы в развёрнутом окне
    (B, 4). Разрешены ходы туда, где у своей клетки виден проход.

    Окно берётся прямо из памяти, уже развёрнутым: при отражении номера строк (столбцов)
    окна идут в обратном порядке, а биты проходов N<->S (E<->W) меняются местами. Всё
    поле целиком не перебирается - только (2K+1)^2 клеток на лабиринт. padded - те же
    known, cells, visits с полями K (BatchEnv держит их готовыми), rows - какие строки
    пачки padded брать (по умолчанию все по порядку)."""
    B = len(pos)
    kp, mp, vp = padded if padded is not None else (pad(known), pad(cells), pad(visits))
    dy = exit_[:, 0] - pos[:, 0]
    dx = exit_[:, 1] - pos[:, 1]
    flip_v = dy < 0
    flip_h = dx < 0
    flips = np.stack([flip_v, flip_h], axis=1)
    off = np.arange(SIDE)
    rr = pos[:, 0, None] + np.where(flip_v[:, None], SIDE - 1 - off, off)    # строки с полями
    cc = pos[:, 1, None] + np.where(flip_h[:, None], SIDE - 1 - off, off)
    b = (np.arange(B) if rows is None else np.asarray(rows))[:, None, None]
    R, Cc = rr[:, :, None], cc[:, None, :]
    kw = kp[b, R, Cc]
    mw = np.where(kw, mp[b, R, Cc], 0)
    vw = vp[b, R, Cc]
    real_r, real_c = R - K, Cc - K
    crop = np.empty((B, SIDE, SIDE, C), np.float32)
    crop[..., 0] = ((real_r >= 0) & (real_r < hw[:, 0, None, None]) &
                    (real_c >= 0) & (real_c < hw[:, 1, None, None]))
    crop[..., 1] = kw
    fv, fh = flip_v[:, None, None], flip_h[:, None, None]
    n, e, s_, w = (mw & 1) > 0, (mw & 2) > 0, (mw & 4) > 0, (mw & 8) > 0
    crop[..., 2] = np.where(fv, s_, n)
    crop[..., 3] = np.where(fh, w, e)
    crop[..., 4] = np.where(fv, n, s_)
    crop[..., 5] = np.where(fh, e, w)
    crop[..., 6] = vw > 0
    crop[..., 7] = np.minimum(vw, 4) / 4.0
    crop[..., 8] = (real_r == exit_[:, 0, None, None]) & (real_c == exit_[:, 1, None, None])

    g = np.zeros((B, GLOBAL), np.float32)
    g[:, 0] = np.minimum(np.abs(dy) / 16.0, 3.0)
    g[:, 1] = np.minimum(np.abs(dx) / 16.0, 3.0)
    g[:, 2] = dy == 0
    g[:, 3] = dx == 0
    has_last = last >= 0
    if has_last.any():
        lc = from_real(np.where(has_last, last, 0), flips)
        g[np.arange(B)[has_last], 4 + lc[has_last]] = 1.0
    legal = crop[:, K, K, 2:6] > 0          # своя клетка всегда видна: её проходы в окне
    x = np.concatenate([crop.reshape(B, -1), g], axis=1)
    return x, flips, legal


def from_real(d, flips):
    """Настоящее направление -> направление в развёрнутом окне (и обратно - отражения
    сами себе обратны)."""
    d = np.asarray(d)
    d = np.where(flips[:, 0], _FLIP_V[d], d)
    d = np.where(flips[:, 1], _FLIP_H[d], d)
    return d


to_real = from_real


def encode_memory(mem):
    """Вход сети для одной памяти fog.Memory (пачка из одного)."""
    return encode(mem.known[None], mem.cells[None], mem.visits[None],
                  np.array([mem.pos]), np.array([mem.exit]), np.array([[mem.h, mem.w]]),
                  np.array([mem.last]))
