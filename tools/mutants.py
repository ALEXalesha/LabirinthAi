# Мутанты для tools/mutate.py: (имя, файл, было, стало, проверки).
G = "game/core/generators.py"
TG = "tests/test_generators.py"
FOG = "game/core/fog.py"
FE = "game/ai/features.py"
TE = "tests/test_env_features.py"
TH = "tests/test_honesty.py"
TS = "tests/test_solvers.py"
MUTANTS = [
    ("gen: Краскал без проверки кусков", G, "        if a != b:\n            parent[a] = b", "        if True:\n            parent[a] = b", TG),
    ("gen: Уилсон без стирания петель", G, "            step[(r, c)] = d", "            step.setdefault((r, c), d)", TG),
    ("gen: деление без двери", G, "                if c - c0 != door:", "                if True:", TG),
    ("gen: Прим без связи", G, "        maze.carve(r, c, rng.choice(links))", "        pass", TG),
    ("gen: петли сносят стены", G, "            maze.carve(r, c, rng.choice(dead_next or closed))",
     "            maze.wall(r, c, rng.choice(_inside_dirs(maze, r, c)))", TG),
    ("fog: видно 5x5", FOG, "\nVIEW = 1\n", "\nVIEW = 2\n", TS),
    ("fog: видно весь лабиринт", FOG, "        m.cells[r0:r1, c0:c1] = self._maze.cells[r0:r1, c0:c1]",
     "        m.cells[r0:r1, c0:c1] = self._maze.cells[r0:r1, c0:c1]\n        m.cells[...] = self._maze.cells", TS + " " + TH),
    ("fog: шаг сквозь стену", FOG, "        if m.done or not self._maze.is_open(*m.pos, d):", "        if m.done:", TS),
    ("features: N/S не меняются при отражении", FE, "    crop[..., 2] = np.where(fv, s_, n)", "    crop[..., 2] = np.where(fv, n, s_)", TE),
    ("features: окно и обзор не отражаются", FE, "    rr = pos[:, 0, None] + np.where(flip_v[:, None], rev, off)",
     "    rr = pos[:, 0, None] + off", TE),
    ("features: невиданные проходы видны", FE, "    mw = np.where(kw, mp[b, R, Cc], 0)", "    mw = mp[b, R, Cc]", TE),
    ("features: выход в окне не на месте", FE,
     "    crop[..., 8] = (real_r == exit_[:, 0, None, None]) & (real_c == exit_[:, 1, None, None])",
     "    crop[..., 8] = (real_r == exit_[:, 0, None, None]) & (real_c == exit_[:, 0, None, None])", TE),
    ("features: обзор - пройденное как виденное", FE, "    stack[..., 2] = vp[b, R, Cc] > 0", "    stack[..., 2] = kp[b, R, Cc]", TE),
    ("features: обзор без выхода", FE, "    out[..., 3] = blocks[..., 3].max(axis=(2, 4))", "    out[..., 3] = 0", TE),
]
