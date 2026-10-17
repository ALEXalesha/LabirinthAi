# LabirinthAi

**Mazes in the fog and an AI that learns to solve mazes it has never seen.** Work in progress (0.1.0).

[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-D97757?logo=claude&logoColor=white)](https://claude.com/claude-code)
Code, tests and docs are written by Claude (Anthropic) in Claude Code. Ideas, testing and decisions - Alexey.

[Русская версия](README.ru.md) · MIT License

- Six seeded generators: depth-first, Prim, Kruskal, Wilson, binary tree, rooms (recursive division); optional loops. Law tests: every maze is connected, perfect mazes have exactly cells − 1 passages.
- Fog: you and the AI see only the 3×3 cells around you plus what you have already seen. Honesty law test: changing an unseen part of the maze never changes the AI's moves.
- Classic solvers for comparison: right hand, Trémaux, an optimistic planner on memory, and BFS / A* (cheaters that see the whole maze).
- A numpy policy network (9×9 memory window + exit direction) trained with PPO on thousands of random mazes. Training in progress.
- PySide6 window: solve it yourself, race the AI, watch several agents explore, AI thoughts, Paint themes, Russian and English.

```
pip install -r requirements.txt
python main.py
python -m pytest -q
python -m game.train --trial      # timing run and forecast before a full training
```
