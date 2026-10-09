"""Журнал обучения: ограничение потоков, прогресс в файл, снимки, продолжение с места.

Обучение идёт на ПК, где работают и другие программы, поэтому:
- потоков BLAS столько, сколько задано (limit_threads зовётся ДО импорта numpy - иначе
  библиотека уже запущена со своим числом потоков);
- каждая строка прогресса сразу пишется в файл (видно, что идёт, даже из другой программы);
- снимок состояния пишется атомарно (временный файл и замена), прерванное обучение
  продолжается с последнего снимка.
"""
import json
import os
import time
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_threads(n, force=False):
    """Число потоков BLAS для этого процесса и его детей."""
    for var in THREAD_VARS:
        if force:
            os.environ[var] = str(n)
        else:
            os.environ.setdefault(var, str(n))


class TrainLog:
    """Строки прогресса в файл (и в консоль, если echo)."""

    def __init__(self, path, echo=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.started = time.time()

    def __call__(self, text):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}"
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


def save_snapshot(path, arrays, meta):
    """arrays - {имя: numpy-массив}, meta - что угодно для JSON. Атомарно."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    os.replace(tmp, path)


def load_snapshot(path):
    """(arrays, meta) или None, если снимка нет или он битый."""
    import numpy as np
    try:
        with np.load(os.fspath(path)) as data:
            arrays = {k: data[k] for k in data.files if k != "__meta__"}
            meta = json.loads(str(data["__meta__"]))
        return arrays, meta
    except (OSError, ValueError, KeyError, EOFError):
        return None
