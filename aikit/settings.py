"""Настройки в JSON по схеме. Запись сразу при изменении; каждое плохое поле в файле
заменяется значением по умолчанию само по себе, остальные остаются (как в TicTacToeAi)."""
from aikit.storage import is_int, load_json, save_json


def choice(*values):
    return lambda v: _safe_in(v, values)


def int_range(lo, hi):
    return lambda v: is_int(v) and lo <= v <= hi


def boolean():
    return lambda v: isinstance(v, bool)


def _safe_in(v, values):
    try:
        return v in values
    except TypeError:          # нехэшируемое значение
        return False


class Settings:
    """schema: {имя: (значение по умолчанию, проверка)}. Доступ: s["имя"], s.set("имя", v)."""

    def __init__(self, schema, path=None):
        self.schema = dict(schema)
        self.path = path
        self.values = {k: d for k, (d, _) in self.schema.items()}
        if path is not None:
            data = load_json(path)
            if isinstance(data, dict):
                for k, (_, check) in self.schema.items():
                    if k in data and self._valid(check, data[k]):
                        self.values[k] = data[k]

    @staticmethod
    def _valid(check, value):
        try:
            return bool(check(value))
        except (TypeError, ValueError):
            return False

    def __getitem__(self, key):
        return self.values[key]

    def set(self, key, value):
        """Записать значение и сохранить файл. Неверное значение - ValueError, файл не трогается."""
        _, check = self.schema[key]
        if not self._valid(check, value):
            raise ValueError(f"{key}: недопустимое значение {value!r}")
        self.values[key] = value
        return self.save()

    def save(self):
        if self.path is None:
            return False
        return save_json(self.path, self.values)
