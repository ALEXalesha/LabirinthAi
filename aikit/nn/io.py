import os

import numpy as np


def save(path, arrays):
    """Атомарно: сначала во временный файл, потом замена - прерванная запись не портит
    прежнюю модель."""
    path = os.fspath(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load(path):
    with np.load(os.fspath(path)) as data:
        return {k: data[k] for k in data.files}


def collect(prefix_layers):
    """{'имя.W': массив} по словарю {'имя': слой} - для сохранения всей сети."""
    return {f"{name}.{p}": arr for name, layer in prefix_layers.items()
            for p, arr, _ in layer.params()}


def restore(prefix_layers, arrays):
    for name, layer in prefix_layers.items():
        for p, arr, _ in layer.params():
            key = f"{name}.{p}"
            if arrays[key].shape != arr.shape:
                raise ValueError(f"{key}: в файле {arrays[key].shape}, в сети {arr.shape}")
            arr[...] = arrays[key]
