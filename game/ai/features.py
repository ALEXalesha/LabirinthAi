"""Что видит сеть: окно памяти вокруг себя и направление на выход.

Всё считается пачкой (B лабиринтов сразу) - так одна и та же функция служит и обучению,
и игре (там B = 1). На вход идёт только память игрока (known, cells, visits) и открытые
всем сведения: размер поля, где выход, свой последний шаг.

Окно разворачивается так, чтобы выход всегда был снизу справа (dy >= 0, dx >= 0): сеть
учит одно правило вместо четырёх. Направления сети - в развёрнутом окне; to_real и
from_canon переводят их обратно.

Каналы каждой клетки окна (C = 9): внутри поля; видел; проходы N, E, S, W (у виденных
клеток); был здесь; сколько раз был (до 4, деленное на 4); выход.

Обзор (с версии 2): вокруг игрока 7 x 7 блоков по 3 x 3 клетки (до 10 клеток в каждую
сторону), в каждом блоке - доля клеток поля, доля виденных, доля пройденных и есть ли там
выход. Окно 9 x 9 видит проходы рядом, обзор - где ещё не был и куда тянется лабиринт:
без него сеть в больших лабиринтах кружила по уже пройденному.
"""
import numpy as np

K = 4                       # окно (2K+1) x (2K+1) клеток вокруг игрока
SIDE = 2 * K + 1
C = 9
BLOCK = 3                   # обзор: блоки BLOCK x BLOCK клеток
GRID = 7                    # GRID x GRID блоков вокруг игрока
CC = 4                      # каналы блока
REACH = BLOCK * GRID // 2   # 10 клеток в каждую сторону
PAD = max(K, REACH)         # поля массивов памяти
GLOBAL = 8
WINDOW = SIDE * SIDE * C
COARSE = GRID * GRID * CC
SIZE = WINDOW + COARSE + GLOBAL

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
    """Поля шириной PAD вокруг поля (B, H, W): окно и обзор у края не выходят за массив."""
    return np.pad(a, ((0, 0), (PAD, PAD), (PAD, PAD)))


def encode(known, cells, visits, pos, exit_, hw, last, padded=None, rows=None):
    """Вход сети (B, SIZE), отражения flips (B, 2) и разрешённые ходы в развёрнутом окне
    (B, 4). Разрешены ходы туда, где у своей клетки виден проход.

    Окно берётся прямо из памяти, уже развёрнутым: при отражении номера строк (столбцов)
    окна идут в обратном порядке, а биты проходов N<->S (E<->W) меняются местами. Всё
    поле целиком не перебирается - только окно и обзор вокруг. padded - те же known, cells,
    visits с полями PAD (BatchEnv держит их готовыми), rows - какие строки
    пачки padded брать (по умолчанию все по порядку)."""
    B = len(pos)
    kp, mp, vp = padded if padded is not None else (pad(known), pad(cells), pad(visits))
    dy = exit_[:, 0] - pos[:, 0]
    dx = exit_[:, 1] - pos[:, 1]
    flip_v = dy < 0
    flip_h = dx < 0
    flips = np.stack([flip_v, flip_h], axis=1)
    b = (np.arange(B) if rows is None else np.asarray(rows))[:, None, None]
    R, Cc = _grid(pos, flip_v, flip_h, K)
    real_r, real_c = R - PAD, Cc - PAD
    kw = kp[b, R, Cc]
    mw = np.where(kw, mp[b, R, Cc], 0)
    vw = vp[b, R, Cc]
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
    coarse = _coarse(kp, vp, b, pos, exit_, hw, flip_v, flip_h)
    x = np.concatenate([crop.reshape(B, -1), coarse.reshape(B, -1), g], axis=1)
    return x, flips, legal


def _grid(pos, flip_v, flip_h, reach):
    """Индексы (с полями PAD) квадрата клеток +-reach вокруг игрока, уже развёрнутого:
    при отражении строки (столбцы) идут в обратном порядке."""
    side = 2 * reach + 1
    off = np.arange(side) + PAD - reach
    rev = side - 1 - np.arange(side) + PAD - reach
    rr = pos[:, 0, None] + np.where(flip_v[:, None], rev, off)
    cc = pos[:, 1, None] + np.where(flip_h[:, None], rev, off)
    return rr[:, :, None], cc[:, None, :]


def _coarse(kp, vp, b, pos, exit_, hw, flip_v, flip_h):
    """Обзор (B, GRID, GRID, CC): доля поля, виденного, пройденного в каждом блоке; выход."""
    B = len(pos)
    R, Cc = _grid(pos, flip_v, flip_h, REACH)
    real_r, real_c = R - PAD, Cc - PAD
    inside = ((real_r >= 0) & (real_r < hw[:, 0, None, None]) &
              (real_c >= 0) & (real_c < hw[:, 1, None, None]))
    ex = (real_r == exit_[:, 0, None, None]) & (real_c == exit_[:, 1, None, None])
    side = GRID * BLOCK
    stack = np.empty((B, side, side, CC), np.float32)
    stack[..., 0] = inside
    stack[..., 1] = kp[b, R, Cc]
    stack[..., 2] = vp[b, R, Cc] > 0
    stack[..., 3] = ex
    blocks = stack.reshape(B, GRID, BLOCK, GRID, BLOCK, CC)
    out = blocks.mean(axis=(2, 4))
    out[..., 3] = blocks[..., 3].max(axis=(2, 4))
    return out


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
