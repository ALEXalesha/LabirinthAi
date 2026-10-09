"""Кто проходит лабиринт. У каждого reset(memory) и act(memory) -> направление.

Честные (видят только память игрока, fog.Memory):
- RightHand - правило правой руки: держится правой рукой за стену;
- Tremaux - алгоритм Тремо: метки на проходах, по проходу не больше двух раз;
- Optimist - кратчайший путь по памяти, неизвестное считается открытым (для сравнения);
- NetAgent - обученная сеть (game/ai/net.py).

Читеры (видят весь лабиринт, подписаны «для сравнения»): BFS и A*. Их reset получает
сам лабиринт - так их нельзя спутать с честными.
"""
import heapq
import random
from collections import deque

import numpy as np

from game.ai import features as F
from game.core.maze import DC, DR, OPPOSITE

STRONG_TEMPERATURE = 1.0       # подбирается по кривой после обучения
HONEST = ("right_hand", "tremaux", "optimist", "net_random", "net_early", "net_strong")
CHEATERS = ("bfs", "astar")


class RightHand:
    name = "right_hand"
    cheater = False

    def reset(self, mem):
        self.heading = 2 if mem.start[0] == 0 else 0       # от рамки внутрь

    def act(self, mem):
        h = self.heading
        for d in ((h + 1) % 4, h, (h + 3) % 4, (h + 2) % 4):
            if mem.can_move(d):
                self.heading = d
                return d
        raise RuntimeError("клетка без проходов")


class Tremaux:
    """Метка на проходе ставится при каждом проходе по нему. В новую развилку идём по
    непомеченному проходу; пришли новым проходом в уже знакомую клетку - назад; по
    проходу с двумя метками не ходим."""
    name = "tremaux"
    cheater = False

    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, mem):
        self.marks = {}
        self.rng = random.Random(self.seed)

    @staticmethod
    def _key(r, c, d):
        if d in (1, 2):
            return (r, c, d)
        return (r + DR[d], c + DC[d], OPPOSITE[d])

    def mark(self, r, c, d):
        return self.marks.get(self._key(r, c, d), 0)

    def act(self, mem):
        r, c = mem.pos
        opts = mem.moves()
        marks = {d: self.mark(r, c, d) for d in opts}
        came = OPPOSITE[mem.last] if mem.last >= 0 else None
        old_cell = came is not None and any(marks[d] > 0 for d in opts if d != came)
        if came is not None and old_cell and marks.get(came) == 1:
            d = came
        else:
            zero = [d for d in opts if marks[d] == 0]
            if zero:
                d = self.rng.choice(zero)
            else:
                ones = [d for d in opts if marks[d] == 1]
                d = came if came in ones else (ones[0] if ones else self.rng.choice(opts))
        k = self._key(r, c, d)
        self.marks[k] = self.marks.get(k, 0) + 1
        return d


def optimistic_distances(mem):
    """Шагов до выхода по памяти, если всё невиданное открыто. -1 - не дойти."""
    h, w = mem.h, mem.w
    dist = np.full((h, w), -1, np.int32)
    dist[mem.exit] = 0
    q = deque([mem.exit])
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for d in range(4):
            r2, c2 = r + DR[d], c + DC[d]
            if 0 <= r2 < h and 0 <= c2 < w and dist[r2, c2] < 0 and mem.edge(r, c, d) != 0:
                dist[r2, c2] = nd
                q.append((r2, c2))
    return dist


class Optimist:
    """Идёт по кратчайшему пути на выход, считая невиданное открытым; увидел стену -
    пересчитал. Честный (только память), но это не обучение, а готовый алгоритм."""
    name = "optimist"
    cheater = False

    def reset(self, mem):
        pass

    def act(self, mem):
        dist = optimistic_distances(mem)
        r, c = mem.pos
        best = None
        for d in mem.moves():
            r2, c2 = r + DR[d], c + DC[d]
            key = (dist[r2, c2] if dist[r2, c2] >= 0 else 1 << 30, mem.visits[r2, c2], d)
            if best is None or key < best[0]:
                best = (key, d)
        self.dist = dist
        return best[1]


class Cheater:
    """BFS или A* по всему лабиринту: идёт ровно по кратчайшему пути. В «Смотреть»
    видно, сколько клеток поиск перебрал (explored)."""
    cheater = True

    def __init__(self, kind="astar"):
        self.name = kind

    def reset(self, mem, maze=None):
        if maze is None:
            raise ValueError("читеру нужен весь лабиринт")
        self.path, self.explored = search(maze, self.name)
        self.i = 0

    def act(self, mem):
        a, b = self.path[self.i], self.path[self.i + 1]
        self.i += 1
        for d in range(4):
            if (a[0] + DR[d], a[1] + DC[d]) == b:
                return d
        raise RuntimeError("путь разорван")


def search(maze, kind="astar"):
    """(путь, порядок перебора клеток) - BFS или A* с манхэттенским расстоянием."""
    start, goal = maze.start, maze.exit
    prev = {start: None}
    order = []
    if kind == "bfs":
        q = deque([start])
        while q:
            cur = q.popleft()
            order.append(cur)
            if cur == goal:
                break
            for _, r2, c2 in maze.neighbors(*cur):
                if (r2, c2) not in prev:
                    prev[(r2, c2)] = cur
                    q.append((r2, c2))
    else:
        def hfun(p):
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        g = {start: 0}
        heap = [(hfun(start), 0, start)]
        done = set()
        while heap:
            _, gc, cur = heapq.heappop(heap)
            if cur in done:
                continue
            done.add(cur)
            order.append(cur)
            if cur == goal:
                break
            for _, r2, c2 in maze.neighbors(*cur):
                nxt = (r2, c2)
                if gc + 1 < g.get(nxt, 1 << 30):
                    g[nxt] = gc + 1
                    prev[nxt] = cur
                    heapq.heappush(heap, (gc + 1 + hfun(nxt), gc + 1, nxt))
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1], order


class NetAgent:
    """Сеть выбирает ход по памяти: случайно по своим вероятностям с зерном (temperature < 1 -
    увереннее), greedy - всегда самый вероятный. Жадная сеть зацикливается: в одинаковой
    памяти один и тот же ход, а счётчик посещений в окне ограничен 4."""
    cheater = False

    def __init__(self, net, name="net_strong", greedy=False, seed=0, temperature=1.0):
        self.net = net
        self.name = name
        self.greedy = greedy
        self.seed = seed
        self.temperature = temperature

    def reset(self, mem):
        self.rng = np.random.default_rng(self.seed)
        self.last_probs = None
        self.last_value = None

    def think(self, mem):
        """(вероятности ходов N E S W в настоящих направлениях, оценка)."""
        x, flips, legal = F.encode_memory(mem)
        p, v = self.net.probs(x, legal)
        real = np.zeros(4)
        for d in range(4):
            real[F.to_real(np.array([d]), flips)[0]] = p[0, d]
        return real, float(v[0])

    def act(self, mem):
        p, v = self.think(mem)
        self.last_probs, self.last_value = p, v
        if self.greedy:
            return int(np.argmax(p))
        if self.temperature != 1.0:
            p = np.where(p > 0, p, 0) ** (1.0 / self.temperature)
        return int(self.rng.choice(4, p=p / p.sum()))


def make_agent(name, seed=0):
    """Агент по имени. Сети берутся из game/models; нет модели - None."""
    from game.ai.net import EARLY, STRONG, PolicyNet, load_model
    if name == "right_hand":
        return RightHand()
    if name == "tremaux":
        return Tremaux(seed)
    if name == "optimist":
        return Optimist()
    if name in CHEATERS:
        return Cheater(name)
    if name == "net_random":
        return NetAgent(PolicyNet(seed=12345), name, greedy=False, seed=seed)
    if name == "net_early":
        net = load_model(EARLY)
        return NetAgent(net, name, greedy=False, seed=seed) if net is not None else None
    if name == "net_strong":
        net = load_model(STRONG)
        return NetAgent(net, name, seed=seed, temperature=STRONG_TEMPERATURE) if net is not None else None
    raise ValueError(f"нет агента {name!r}")


def level_agent(level, seed=0):
    """Уровни: новичок - правая рука, средний - ранняя сеть, сильный - обученная.
    Пока сети нет, средний - Тремо, сильный - Оптимист (видно в окне)."""
    if level == "novice":
        return RightHand()
    if level == "medium":
        return make_agent("net_early", seed) or Tremaux(seed)
    if level == "strong":
        return make_agent("net_strong", seed) or Optimist()
    raise ValueError(level)

