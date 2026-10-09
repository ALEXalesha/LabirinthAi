"""Темы Paint: тёмная и светлая. Цвета для рисования и таблица стилей QSS.

Значения взяты из тем Paint Pro (paint-pro-electron/paint-pro.html, блоки :root[data-theme]):
тёмная - «Строгая» (formal), светлая - «Светлая» (light). Стекло Paint (полупрозрачные слои
поверх фона) здесь сведено к сплошным цветам: Qt без композиции окна его не нарисует,
поэтому surface и surface2 - это стекло, смешанное с фоном темы.

Сверх токенов Paint - цвета досок (клетки, подсветки) и шкала «Мыслей ИИ».
"""

PALETTES = {
    "dark": {
        # Paint «Строгая»: --app-base #1f242e, --text #e6e8ec, --accent #4a7ab8
        "bg": "#1f242e", "surface": "#272c36", "surface2": "#2f3540", "border": "#3a404b",
        "text": "#e6e8ec", "muted": "#9a9fa8", "accent": "#4a7ab8", "accent2": "#5c7fa6",
        "accent_text": "#ffffff", "good": "#2dd4a8", "bad": "#ff5b5b", "warn": "#e8b04a",
        "grid": "#333946",
        "sq_light": "#c9d1dd", "sq_dark": "#56688a", "coord": "#9aa6ba",
        "last_move": "#e8c45a", "select": "#5fb4ff", "hint": "#1b1f27", "capture_hint": "#ff8a5b",
        "piece_white": "#f4f1ea", "piece_white_edge": "#3b3f47",
        "piece_black": "#2a2e36", "piece_black_edge": "#cfd4dc",
    },
    "light": {
        # Paint «Светлая»: --app-base #e8ecf3, --text #1b2030, --accent #3a6fd8
        "bg": "#e8ecf3", "surface": "#f6f7fa", "surface2": "#dfe4ec", "border": "#c6cdd9",
        "text": "#1b2030", "muted": "#5f6575", "accent": "#3a6fd8", "accent2": "#7a52d0",
        "accent_text": "#ffffff", "good": "#169c7a", "bad": "#d8404f", "warn": "#b07a10",
        "grid": "#d5dbe5",
        "sq_light": "#eef1f6", "sq_dark": "#8ba0c4", "coord": "#5f6f8c",
        "last_move": "#f0c43c", "select": "#2a7de1", "hint": "#1b2030", "capture_hint": "#e0603a",
        "piece_white": "#fbfaf6", "piece_white_edge": "#2c3140",
        "piece_black": "#252a35", "piece_black_edge": "#1b2030",
    },
}

THEMES = tuple(PALETTES)


def palette(name):
    return PALETTES.get(name, PALETTES["dark"])


def heat(value, name="dark"):
    """Цвет оценки от -1 (красный) через нейтральный к +1 (зелёный), как в «Мыслях сети»
    TicTacToeAi. Возвращает (r, g, b) - 0..255."""
    v = max(-1.0, min(1.0, float(value)))
    p = palette(name)
    mid = _rgb(p["surface2"])
    end = _rgb(p["good"] if v >= 0 else p["bad"])
    t = abs(v)
    return tuple(round(m + (e - m) * t) for m, e in zip(mid, end))


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def qss(name):
    p = palette(name)
    return f"""
* {{
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: 14px;
    color: {p['text']};
}}
QMainWindow, QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget#page, QStackedWidget {{
    background: {p['bg']};
}}
QScrollArea {{ border: none; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QLabel#title {{ font-size: 38px; font-weight: 700; }}
QLabel#heading {{ font-size: 26px; font-weight: 700; }}
QLabel#section {{ font-size: 16px; font-weight: 600; }}
QLabel#muted, QLabel#hint {{ color: {p['muted']}; }}
QLabel#hint {{ font-size: 12px; }}
QLabel#status {{ font-size: 20px; font-weight: 700; }}
QLabel#statusWin {{ font-size: 20px; font-weight: 700; color: {p['good']}; }}
QLabel#statusLoss {{ font-size: 20px; font-weight: 700; color: {p['bad']}; }}
QLabel#statusDraw {{ font-size: 20px; font-weight: 700; color: {p['warn']}; }}
QLabel#bigNumber {{ font-size: 26px; font-weight: 700; }}
QFrame#card {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 14px;
}}
QPushButton {{
    background: {p['surface2']};
    border: 1px solid {p['border']};
    border-radius: 10px;
    padding: 8px 16px;
}}
QPushButton:hover {{ border-color: {p['accent']}; }}
QPushButton:pressed {{ background: {p['grid']}; }}
QPushButton:disabled {{ color: {p['muted']}; }}
QPushButton:focus {{ outline: none; }}
QPushButton#primary, QPushButton#menuPrimary {{
    background: {p['accent']};
    color: {p['accent_text']};
    border: 1px solid {p['accent']};
    font-weight: 600;
}}
QPushButton#primary:hover, QPushButton#menuPrimary:hover {{ background: {p['accent2']}; border-color: {p['accent2']}; }}
QPushButton#menu, QPushButton#menuPrimary {{
    font-size: 17px;
    padding: 12px 20px;
    min-width: 240px;
}}
QPushButton#seg {{
    border-radius: 8px;
    padding: 6px 10px;
    background: {p['surface2']};
}}
QPushButton#seg:checked {{
    background: {p['accent']};
    color: {p['accent_text']};
    border-color: {p['accent']};
    font-weight: 600;
}}
QComboBox {{
    background: {p['surface2']};
    border: 1px solid {p['border']};
    border-radius: 8px;
    padding: 6px 30px 6px 10px;
    min-width: 140px;
}}
QComboBox:hover {{ border-color: {p['accent']}; }}
QComboBox::drop-down {{ border: none; background: transparent; width: 26px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
QComboBox QAbstractItemView {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    selection-background-color: {p['accent']};
    selection-color: {p['accent_text']};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {p['surface2']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {p['accent']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {p['text']};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid {p['border']};
    background: {p['surface2']};
}}
QCheckBox::indicator:checked {{
    background: {p['accent']};
    border-color: {p['accent']};
    image: none;
}}
QListWidget, QTableWidget {{
    background: {p['surface']};
    border: none;
    gridline-color: {p['grid']};
    selection-background-color: {p['surface2']};
    selection-color: {p['text']};
}}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{
    background: {p['surface']};
    color: {p['muted']};
    border: none;
    border-bottom: 1px solid {p['border']};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {p['surface']}; border: none; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {p['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}
QToolTip {{
    background: {p['surface']};
    color: {p['text']};
    border: 1px solid {p['border']};
}}
"""
