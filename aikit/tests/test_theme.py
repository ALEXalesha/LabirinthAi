import pytest

from aikit import theme


def test_both_themes_have_the_same_keys():
    keys = [set(p) for p in theme.PALETTES.values()]
    assert all(k == keys[0] for k in keys)


def test_themes_take_paint_values():
    # «Строгая» и «Светлая» Paint: фон, текст и акцент - строка в строку
    d, li = theme.PALETTES["dark"], theme.PALETTES["light"]
    assert (d["bg"], d["text"], d["accent"]) == ("#1f242e", "#e6e8ec", "#4a7ab8")
    assert (li["bg"], li["text"], li["accent"]) == ("#e8ecf3", "#1b2030", "#3a6fd8")


def _lum(hex_color):
    def ch(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = theme._rgb(hex_color)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("name", theme.THEMES)
def test_text_is_readable(name):
    p = theme.palette(name)
    for bg in ("bg", "surface", "surface2"):
        assert contrast(p["text"], p[bg]) >= 7, bg
    assert contrast(p["muted"], p["surface"]) >= 4.5
    assert contrast(p["accent_text"], p["accent"]) >= 3


@pytest.mark.parametrize("name", theme.THEMES)
def test_pieces_stand_out_on_both_square_colors(name):
    p = theme.palette(name)
    assert contrast(p["piece_white_edge"], p["piece_white"]) >= 3
    for sq in ("sq_light", "sq_dark"):
        assert max(contrast(p["piece_black"], p[sq]), contrast(p["piece_black_edge"], p[sq])) >= 1.6
        assert max(contrast(p["piece_white"], p[sq]), contrast(p["piece_white_edge"], p[sq])) >= 1.6


def test_heat_runs_from_bad_through_neutral_to_good():
    p = theme.palette("dark")
    assert theme.heat(1, "dark") == theme._rgb(p["good"])
    assert theme.heat(-1, "dark") == theme._rgb(p["bad"])
    assert theme.heat(0, "dark") == theme._rgb(p["surface2"])
    assert theme.heat(7, "dark") == theme.heat(1, "dark")      # за краем шкалы - край


def test_unknown_theme_falls_back_to_dark():
    assert theme.palette("нет такой") is theme.PALETTES["dark"]


@pytest.mark.parametrize("name", theme.THEMES)
def test_combo_has_rounded_corners_and_a_drawn_arrow(qapp, name):
    """Ловушка AiCar 2.0.0: без ::drop-down Fusion рисует квадратную кнопку поверх скругления."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from aikit.widgets import Combo
    p = theme.palette(name)
    host = QWidget()
    host.setObjectName("page")
    host.setStyleSheet(theme.qss(name))
    lay = QVBoxLayout(host)
    combo = Combo()
    combo.set_arrow_color(p["text"])
    combo.addItems(["Первый вариант", "Второй"])
    lay.addWidget(combo)
    host.resize(260, 80)
    host.show()
    qapp.processEvents()
    img = host.grab().toImage()
    g = combo.geometry()
    page = QColor(p["bg"])
    # все четыре угла за радиусом - фон страницы, не квадрат кнопки поверх скругления
    for x, y in ((g.x(), g.y()), (g.right(), g.y()), (g.x(), g.bottom()), (g.right(), g.bottom())):
        corner = img.pixelColor(x, y)
        assert abs(corner.red() - page.red()) + abs(corner.green() - page.green()) +             abs(corner.blue() - page.blue()) < 40, (x, y)
    text = QColor(p["text"])

    def near_text(c):
        return abs(c.red() - text.red()) + abs(c.green() - text.green()) + abs(c.blue() - text.blue()) < 90

    def row_width(y):
        return sum(1 for x in range(g.right() - 30, g.right()) if near_text(img.pixelColor(x, y)))

    cy = g.y() + g.height() // 2
    widths = [row_width(y) for y in range(cy - 6, cy + 7)]
    assert max(widths) >= 6
    top = next(i for i, v in enumerate(widths) if v)
    bottom = len(widths) - 1 - next(i for i, v in enumerate(reversed(widths)) if v)
    assert widths[top] > widths[bottom]
    # у кнопки ::drop-down нет своего тёмного квадрата (у Fusion он темнее страницы)
    darkest = min(img.pixelColor(x, y).lightness()
                  for x in range(g.right() - 26, g.right()) for y in range(g.y() + 2, g.bottom()))
    assert darkest >= min(QColor(p["surface2"]).lightness(), text.lightness()) - 25
    host.close()
