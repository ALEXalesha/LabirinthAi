"""Генераторы лабиринтов. Одно зерно - один и тот же лабиринт.

Каждый генератор строит «идеальный» лабиринт: между любыми двумя клетками ровно один
путь. Значит, все клетки связаны и проходов ровно на один меньше, чем клеток (законы в
tests/test_generators.py). Петли добавляет отдельно add_loops: убирает часть тупиков.

    generate("dfs", 15, 15, seed=7)
    generate("kruskal", 12, 20, seed=1, loops=0.3)
"""
import random

from game.core.maze import BIT, DC, DIRS, DR, E, N, OPPOSITE, S, W, Maze

KINDS = ("dfs", "prim", "kruskal", "wilson", "binary", "rooms")


def generate(kind, h, w, seed=0, loops=0.0):
    if kind not in GENERATORS:
        raise ValueError(f"нет генератора {kind!r}")
    if h < 2 or w < 2:
        raise ValueError("лабиринт меньше 2 x 2")
    rng = random.Random(f"{kind}:{h}x{w}:{seed}")
    maze = Maze.closed(h, w, kind=kind, seed=seed)
    GENERATORS[kind](maze, rng)
    if loops > 0:
        add_loops(maze, loops, rng)
    return maze


def _inside_dirs(maze, r, c):
    return [d for d in DIRS if 0 <= r + DR[d] < maze.h and 0 <= c + DC[d] < maze.w]


def dfs(maze, rng):
    """Поиск в глубину с возвратом: длинные извилистые коридоры, мало развилок."""
    seen = [[False] * maze.w for _ in range(maze.h)]
    r, c = rng.randrange(maze.h), rng.randrange(maze.w)
    seen[r][c] = True
    stack = [(r, c)]
    while stack:
        r, c = stack[-1]
        options = [d for d in _inside_dirs(maze, r, c) if not seen[r + DR[d]][c + DC[d]]]
        if not options:
            stack.pop()
            continue
        d = rng.choice(options)
        maze.carve(r, c, d)
        r2, c2 = r + DR[d], c + DC[d]
        seen[r2][c2] = True
        stack.append((r2, c2))


def prim(maze, rng):
    """Случайный Прим: растёт от одной клетки во все стороны, много коротких тупиков."""
    inside = [[False] * maze.w for _ in range(maze.h)]
    r, c = rng.randrange(maze.h), rng.randrange(maze.w)
    inside[r][c] = True
    frontier = []
    in_frontier = set()

    def add(r, c):
        for d in _inside_dirs(maze, r, c):
            r2, c2 = r + DR[d], c + DC[d]
            if not inside[r2][c2] and (r2, c2) not in in_frontier:
                in_frontier.add((r2, c2))
                frontier.append((r2, c2))

    add(r, c)
    while frontier:
        i = rng.randrange(len(frontier))
        frontier[i], frontier[-1] = frontier[-1], frontier[i]
        r, c = frontier.pop()
        in_frontier.discard((r, c))
        links = [d for d in _inside_dirs(maze, r, c) if inside[r + DR[d]][c + DC[d]]]
        maze.carve(r, c, rng.choice(links))
        inside[r][c] = True
        add(r, c)


def kruskal(maze, rng):
    """Случайный Краскал: стены в случайном порядке, сносится та, что соединяет разные
    куски (система непересекающихся множеств)."""
    parent = list(range(maze.h * maze.w))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = [(r, c, E) for r in range(maze.h) for c in range(maze.w - 1)]
    edges += [(r, c, S) for r in range(maze.h - 1) for c in range(maze.w)]
    rng.shuffle(edges)
    for r, c, d in edges:
        a = find(r * maze.w + c)
        b = find((r + DR[d]) * maze.w + c + DC[d])
        if a != b:
            parent[a] = b
            maze.carve(r, c, d)


def wilson(maze, rng):
    """Уилсон: случайные блуждания со стиранием петель. Каждый идеальный лабиринт
    выходит с равной вероятностью - без перекоса к длинным коридорам или тупикам."""
    h, w = maze.h, maze.w
    inside = [[False] * w for _ in range(h)]
    inside[rng.randrange(h)][rng.randrange(w)] = True
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    for start in cells:
        if inside[start[0]][start[1]]:
            continue
        step = {}                    # клетка -> направление, которым из неё ушли последний раз
        r, c = start
        while not inside[r][c]:
            d = rng.choice(_inside_dirs(maze, r, c))
            step[(r, c)] = d
            r, c = r + DR[d], c + DC[d]
        r, c = start
        while not inside[r][c]:
            d = step[(r, c)]
            maze.carve(r, c, d)
            inside[r][c] = True
            r, c = r + DR[d], c + DC[d]


def binary(maze, rng):
    """«Двоичное дерево»: из каждой клетки проход на север или на запад. Верхний ряд и
    левый столбец - сплошные коридоры, лабиринт перекошен к левому верхнему углу."""
    for r in range(maze.h):
        for c in range(maze.w):
            options = [d for d in (N, W) if 0 <= r + DR[d] and 0 <= c + DC[d]]
            if options:
                maze.carve(r, c, rng.choice(options))


def rooms(maze, rng):
    """Рекурсивное деление: поле режется стенами с одной дверью, куски - снова, до
    коридоров в одну клетку. Длинные прямые стены, похоже на комнаты."""
    h, w = maze.h, maze.w
    for r in range(h):
        for c in range(w):
            if c + 1 < w:
                maze.cells[r, c] |= BIT[E]
                maze.cells[r, c + 1] |= BIT[W]
            if r + 1 < h:
                maze.cells[r, c] |= BIT[S]
                maze.cells[r + 1, c] |= BIT[N]
    stack = [(0, 0, h, w)]
    while stack:
        r0, c0, hh, ww = stack.pop()
        if hh < 2 or ww < 2:
            continue
        horizontal = hh > ww or (hh == ww and rng.random() < 0.5)
        if horizontal:
            cut = rng.randrange(1, hh)                 # стена между строками r0+cut-1 и r0+cut
            door = rng.randrange(ww)
            for c in range(c0, c0 + ww):
                if c - c0 != door:
                    maze.wall(r0 + cut - 1, c, S)
            stack += [(r0, c0, cut, ww), (r0 + cut, c0, hh - cut, ww)]
        else:
            cut = rng.randrange(1, ww)
            door = rng.randrange(hh)
            for r in range(r0, r0 + hh):
                if r - r0 != door:
                    maze.wall(r, c0 + cut - 1, E)
            stack += [(r0, c0, hh, cut), (r0, c0 + cut, hh, ww - cut)]


GENERATORS = {"dfs": dfs, "prim": prim, "kruskal": kruskal, "wilson": wilson,
              "binary": binary, "rooms": rooms}


def add_loops(maze, share, rng):
    """Петли: share от числа тупиков (0..1) получают второй выход - в соседа, с которым
    прохода ещё нет. Лабиринт перестаёт быть идеальным, но остаётся связным."""
    dead = [(r, c) for r in range(maze.h) for c in range(maze.w) if bin(int(maze.cells[r, c])).count("1") == 1]
    rng.shuffle(dead)
    k = round(len(dead) * max(0.0, min(1.0, share)))
    for r, c in dead[:k]:
        if bin(int(maze.cells[r, c])).count("1") != 1:
            continue                        # уже не тупик: открыт соседним тупиком
        closed = [d for d in _inside_dirs(maze, r, c) if not maze.cells[r, c] & BIT[d]]
        if closed:
            # лучше в другой тупик - так исчезают оба
            dead_next = [d for d in closed
                         if bin(int(maze.cells[r + DR[d], c + DC[d]])).count("1") == 1]
            maze.carve(r, c, rng.choice(dead_next or closed))


def opposite_corners(h, w, rng):
    """Вход и выход в случайной паре противоположных углов."""
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    a = rng.randrange(4)
    return corners[a], corners[3 - a]


__all__ = ["KINDS", "generate", "add_loops", "opposite_corners", "OPPOSITE"]
