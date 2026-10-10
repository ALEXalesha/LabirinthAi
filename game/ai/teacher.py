"""Учитель для подражания: Оптимист, посчитанный сразу на пачку лабиринтов.

Оптимист (agents.Optimist) идёт по кратчайшему пути к выходу по своей памяти, считая
невиданное открытым. Он честный: видит только память. Здесь то же самое массивами numpy -
расстояния до выхода волной (повторять «сосед + 1», пока что-то меняется) по всей пачке.
Закон в тестах: на тех же памятях выбор совпадает с одиночным Оптимистом ход в ход.
"""
import numpy as np

BIG = 1 << 29


def passable(known, mem, hw):
    """Проходы по памяти (B, H, W, 4): видна стена - нет, иначе (открыт или не виден) - да;
    за рамку поля - нет."""
    B, H, W = known.shape
    rows = np.arange(H)[None, :, None]
    cols = np.arange(W)[None, None, :]
    h, w = hw[:, 0, None, None], hw[:, 1, None, None]
    out = np.zeros((B, H, W, 4), bool)
    # стена между a и b видна, если видна клетка a (её бит) или клетка b (её встречный бит)
    for d, (dr, dc, bit, opp) in enumerate(((-1, 0, 1, 4), (0, 1, 2, 8), (1, 0, 4, 1), (0, -1, 8, 2))):
        r2, c2 = rows + dr, cols + dc
        inside = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w) & (rows < h) & (cols < w)
        r2c = np.clip(r2, 0, H - 1)
        c2c = np.clip(c2, 0, W - 1)
        b = np.arange(B)[:, None, None]
        known2 = known[b, r2c, c2c]
        mem2 = mem[b, r2c, c2c]
        wall = (known & ((mem & bit) == 0)) | (known2 & ((mem2 & opp) == 0))
        out[..., d] = inside & ~wall
    return out


def distances(known, mem, hw, exit_, pos=None):
    """Шагов до выхода по памяти, невиданное открыто (B, H, W); BIG - не дойти.

    Волна от выхода: «сосед + 1» по проходам, на месте (обновлённое сразу идёт дальше -
    это только быстрее). После t проходов верны все клетки ближе t к выходу. Если дан pos,
    волна останавливается, как только верны соседи игрока: их расстояния отличаются от
    лучшего не больше чем на 2 (они соседи через клетку игрока)."""
    B, H, W = known.shape
    P = passable(known, mem, hw)
    cost = [np.where(P[..., d], 1, BIG).astype(np.int32) for d in range(4)]
    dist = np.full((B, H, W), BIG, np.int32)
    b = np.arange(B)
    dist[b, exit_[:, 0], exit_[:, 1]] = 0
    for t in range(1, H * W + 2):
        np.minimum(dist[:, 1:, :], dist[:, :-1, :] + cost[0][:, 1:, :], out=dist[:, 1:, :])
        np.minimum(dist[:, :-1, :], dist[:, 1:, :] + cost[2][:, :-1, :], out=dist[:, :-1, :])
        np.minimum(dist[:, :, :-1], dist[:, :, 1:] + cost[1][:, :, :-1], out=dist[:, :, :-1])
        np.minimum(dist[:, :, 1:], dist[:, :, :-1] + cost[3][:, :, 1:], out=dist[:, :, 1:])
        np.minimum(dist, BIG, out=dist)
        if pos is not None:
            # только соседи за открытым проходом: за стеной может лежать хоть сам выход
            here = mem[b, pos[:, 0], pos[:, 1]]
            m = np.full(B, BIG, np.int64)
            for dr, dc, bit in ((-1, 0, 1), (0, 1, 2), (1, 0, 4), (0, -1, 8)):
                r2 = np.clip(pos[:, 0] + dr, 0, H - 1)
                c2 = np.clip(pos[:, 1] + dc, 0, W - 1)
                m = np.where((here & bit) > 0, np.minimum(m, dist[b, r2, c2]), m)
            if np.all(m + 2 <= t):
                return dist
    return dist


def choose(known, mem, visits, pos, exit_, hw):
    """Ход Оптимиста (настоящие направления) для каждого лабиринта пачки: к соседу с
    меньшим расстоянием, при равных - где реже бывал, потом по номеру направления."""
    B = len(pos)
    dist = distances(known, mem, hw, exit_, pos)
    b = np.arange(B)
    here = mem[b, pos[:, 0], pos[:, 1]]
    best_key = np.full(B, np.inf)
    best = np.full(B, -1, np.int64)
    H, W = known.shape[1:]
    for d, (dr, dc, bit) in enumerate(((-1, 0, 1), (0, 1, 2), (1, 0, 4), (0, -1, 8))):
        ok = (here & bit) > 0
        r2 = np.clip(pos[:, 0] + dr, 0, H - 1)
        c2 = np.clip(pos[:, 1] + dc, 0, W - 1)
        dd = np.minimum(dist[b, r2, c2], 10 ** 6).astype(np.float64)      # не дойти - дальше любого пути
        key = dd * 1e7 + np.minimum(visits[b, r2, c2], 1e6) * 10 + d
        better = ok & (key < best_key)
        best_key = np.where(better, key, best_key)
        best = np.where(better, d, best)
    return best
