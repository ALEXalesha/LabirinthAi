"""Перевод строк интерфейса: словари {ключ: текст} на каждый язык.

Нет строки в языке - берётся русская, нет и её - сам ключ (видно в окне, тест ловит).
"""


class Translator:
    def __init__(self, tables, lang="ru"):
        self.tables = tables
        self.lang = lang

    def __call__(self, key, **kw):
        table = self.tables.get(self.lang) or {}
        text = table.get(key)
        if text is None:
            text = self.tables.get("ru", {}).get(key, key)
        return text.format(**kw) if kw else text

    def missing(self):
        """Ключи, которые есть в одном языке и нет в другом: {язык: [ключи]}."""
        keys = set().union(*(set(t) for t in self.tables.values()))
        return {lang: sorted(keys - set(t)) for lang, t in self.tables.items() if keys - set(t)}
