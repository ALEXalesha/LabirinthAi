# LabirinthAi

**Лабиринты в тумане и ИИ, который учится проходить незнакомые лабиринты.** В работе (0.1.0).

[![Status](https://img.shields.io/badge/status-in_development-orange)](#) [![CI](https://github.com/ALEXalesha/LabirinthAi/actions/workflows/ci.yml/badge.svg)](https://github.com/ALEXalesha/LabirinthAi/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PySide6](https://img.shields.io/badge/PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/) [![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-D97757?logo=claude&logoColor=white)](https://claude.com/claude-code)

Код, тесты и документация этого проекта написаны Claude (Anthropic) в Claude Code. Идеи, проверка и решения - Алексей.

[English version](README.md) · Лицензия MIT

- Шесть генераторов с зерном: глубина (DFS), Прим, Краскал, Уилсон, двоичное дерево, комнаты (деление); по желанию с петлями. Законы-тесты: лабиринт связный, у идеального проходов ровно на один меньше, чем клеток.
- Туман: и вы, и ИИ видите только 3×3 клетки вокруг и то, что уже видели. Закон честности: правка невиданной части лабиринта не меняет ходов ИИ.
- Классика для сравнения: правая рука, Тремо, оптимист по памяти, BFS и A* (читеры, видят весь лабиринт).
- Сеть на numpy (окно памяти 9×9 и направление на выход), обучение PPO на тысячах случайных лабиринтов. Обучение идёт.
- Окно на PySide6: пройти самому, гонка с ИИ, «Смотреть ИИ», «Мысли ИИ», темы Paint, русский и английский.

```
pip install -r requirements.txt
python main.py
python -m pytest -q
python -m game.train --trial      # пробный прогон и прогноз времени перед полным обучением
```
