"""Виджет игры (create_widget по контракту): меню, проход, гонка, «Смотреть ИИ»,
статистика, обучение, настройки.

Окно ему даёт оболочка (aikit.run_standalone или ИИ-Игротека). Тема и язык приходят из
ctx и меняются на лету по ctx.changed. Файлы - только в ctx.data_dir. Потоков нет: ИИ
делает шаг за доли миллисекунды и ходит по таймеру; shutdown останавливает таймеры.
"""
import os
import random

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                               QScrollArea, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from aikit import settings as st
from aikit import theme
from aikit.storage import load_json, save_json
from aikit.widgets import Combo, button, card, label, restyle, segmented
from game.ai.agents import CHEATERS, HONEST, NetAgent
from game.ai.net import MODEL_DIR, STRONG, load_model, load_passport
from game.core.generators import KINDS
from game.ui.board import MazeBoard
from game.ui.chart import CurveChart
from game.ui.i18n import translator
from game.ui.session import SIZES, Session, make_maze
from game.version import VERSION

LEVELS = ("novice", "medium", "strong")
WATCH_AGENTS = HONEST + CHEATERS
WATCH_DEFAULT = ("right_hand", "tremaux", "net_early", "net_strong")
SCHEMA = {
    "size": ("m", st.choice(*SIZES)),
    "kind": ("random", st.choice("random", *KINDS)),
    "loops": (False, st.boolean()),
    "level": ("strong", st.choice(*LEVELS)),
    "show_thoughts": (False, st.boolean()),
    "ai_speed": (6, st.int_range(1, 30)),
    "watch_speed": (12, st.int_range(1, 60)),
    **{f"watch{i}": (WATCH_DEFAULT[i], st.choice(*WATCH_AGENTS)) for i in range(4)},
}
KEYS = {0: ("Up", "W", "Ц"), 1: ("Right", "D", "В"), 2: ("Down", "S", "Ы"), 3: ("Left", "A", "Ф")}
HISTORY_MAX = 500


def page_widget():
    w = QWidget()
    w.setObjectName("page")
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return w


def fmt_time(t, seconds):
    if seconds < 90:
        return t("seconds", n=f"{seconds:.1f}")
    if seconds < 5400:
        return t("minutes", n=round(seconds / 60))
    return t("hours", n=f"{seconds / 3600:.1f}")


def number(n):
    return f"{n:,}".replace(",", " ")


def side_panel(width=330):
    side = card()
    side.setFixedWidth(width)
    outer = QVBoxLayout(side)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
    outer.addWidget(scroll)
    body = QWidget()
    scroll.setWidget(body)
    col = QVBoxLayout(body)
    col.setContentsMargins(16, 14, 16, 14)
    col.setSpacing(8)
    return side, col


# --- меню ---


class MenuPage(QWidget):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        outer = QVBoxLayout(self)
        outer.addStretch(2)
        col = QVBoxLayout()
        col.setSpacing(10)
        col.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QLabel
        from game.ui.icon import render
        self.logo = QLabel()
        self.logo.setPixmap(QPixmap.fromImage(render(104)))
        col.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignHCenter)
        self.title = label("", "title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self.title)
        self.sub = label("", "muted", wrap=True)
        self.sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub.setFixedWidth(640)
        col.addWidget(self.sub, 0, Qt.AlignmentFlag.AlignHCenter)
        col.addSpacing(10)
        box = card()
        box.setMaximumWidth(700)
        grid = QGridLayout(box)
        grid.setContentsMargins(22, 18, 22, 18)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        self.captions = {}

        def row(r, key, widget):
            cap = label("", "muted")
            self.captions[key] = cap
            grid.addWidget(cap, r, 0)
            grid.addWidget(widget, r, 1)

        w, self.size_buttons = segmented({k: "" for k in SIZES}, lambda v: gw.change("size", v))
        row(0, "size", w)
        self.kind_combo = Combo()
        for k in ("random",) + KINDS:
            self.kind_combo.addItem("", k)
        self.kind_combo.currentIndexChanged.connect(lambda i: gw.change("kind", self.kind_combo.itemData(i)))
        row(1, "kind", self.kind_combo)
        w, self.loops_buttons = segmented({False: "", True: ""}, lambda v: gw.change("loops", v))
        row(2, "loops", w)
        w, self.level_buttons = segmented({lv: "" for lv in LEVELS}, lambda v: gw.change("level", v))
        row(3, "level", w)
        col.addWidget(box, 0, Qt.AlignmentFlag.AlignHCenter)
        self.note = label("", "hint", wrap=True)
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note.setFixedWidth(640)
        col.addWidget(self.note, 0, Qt.AlignmentFlag.AlignHCenter)
        col.addSpacing(6)
        self.solo_button = button("", "menuPrimary", lambda: gw.start("solo"))
        self.race_button = button("", "menu", lambda: gw.start("race"))
        self.watch_button = button("", "menu", lambda: gw.start("watch"))
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.stats_button = button("", None, lambda: gw.show_page("stats"))
        self.training_button = button("", None, lambda: gw.show_page("training"))
        self.settings_button = button("", None, lambda: gw.show_page("settings"))
        for b in (self.stats_button, self.training_button, self.settings_button):
            row2.addWidget(b)
        for b in (self.solo_button, self.race_button, self.watch_button):
            col.addWidget(b, 0, Qt.AlignmentFlag.AlignHCenter)
        col.addSpacing(4)
        col.addLayout(row2)
        outer.addLayout(col)
        outer.addStretch(3)
        self.foot = label("", "hint")
        self.foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.foot)

    def sync(self, s):
        self.size_buttons[s["size"]].setChecked(True)
        self.loops_buttons[s["loops"]].setChecked(True)
        self.level_buttons[s["level"]].setChecked(True)
        self.kind_combo.blockSignals(True)
        self.kind_combo.setCurrentIndex(self.kind_combo.findData(s["kind"]))
        self.kind_combo.blockSignals(False)
        self.note.setText("" if load_model(STRONG) is not None else self.gw.t("no_net"))
        self.note.setVisible(bool(self.note.text()))

    def retranslate(self):
        t = self.gw.t
        self.title.setText(t("app_title"))
        self.sub.setText(t("app_sub"))
        for key, cap in self.captions.items():
            cap.setText(t(key))
        for k, b in self.size_buttons.items():
            b.setText(t("size_" + k))
        for i in range(self.kind_combo.count()):
            self.kind_combo.setItemText(i, t("kind_" + self.kind_combo.itemData(i)))
        self.loops_buttons[False].setText(t("loops_off"))
        self.loops_buttons[True].setText(t("loops_on"))
        for k, b in self.level_buttons.items():
            b.setText(t(k))
        self.solo_button.setText(t("mode_solo"))
        self.race_button.setText(t("mode_race"))
        self.watch_button.setText(t("mode_watch"))
        self.stats_button.setText(t("stats"))
        self.training_button.setText(t("training"))
        self.settings_button.setText(t("settings"))
        self.foot.setText(t("footer", v=VERSION))
        self.sync(self.gw.settings)


# --- проход и гонка ---


class PlayPage(QWidget):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        self.boards = [MazeBoard(), MazeBoard()]
        for b in self.boards:
            root.addWidget(b, 1)
        side, col = side_panel()
        self.heading = label("", "section", wrap=True)
        col.addWidget(self.heading)
        self.status = label("", "status", wrap=True)
        col.addWidget(self.status)
        self.substatus = label("", "muted", wrap=True)
        self.substatus.setMinimumHeight(38)     # только снизу: перенос строк не обрезается
        self.substatus.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        col.addWidget(self.substatus)
        self.info = label("", "", wrap=True)
        col.addWidget(self.info)
        col.addSpacing(6)
        self.thoughts_box = QCheckBox("")
        self.thoughts_box.toggled.connect(lambda on: gw.change("show_thoughts", on))
        col.addWidget(self.thoughts_box)
        self.thoughts_hint = label("", "hint", wrap=True)
        col.addWidget(self.thoughts_hint)
        col.addStretch(1)
        self.again_button = button("", "primary", lambda: gw.start(gw.session.mode if gw.session else "solo"))
        self.same_button = button("", None, gw.restart_same)
        self.menu_button = button("", None, lambda: gw.show_page("menu"))
        for b in (self.again_button, self.same_button, self.menu_button):
            col.addWidget(b)
        root.addWidget(side)

    def retranslate(self):
        t = self.gw.t
        self.thoughts_box.setText(t("thoughts"))
        self.again_button.setText(t("again"))
        self.same_button.setText(t("same"))
        self.menu_button.setText(t("menu"))


# --- смотреть ИИ ---


class WatchPage(QWidget):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        grid_box = QWidget()
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.boards = []
        for i in range(4):
            b = MazeBoard()
            grid.addWidget(b, i // 2, i % 2)
            self.boards.append(b)
        root.addWidget(grid_box, 1)
        side, col = side_panel()
        self.heading = label("", "section", wrap=True)
        col.addWidget(self.heading)
        self.hint = label("", "hint", wrap=True)
        col.addWidget(self.hint)
        self.combos = []
        for i in range(4):
            c = Combo()
            for name in WATCH_AGENTS:
                c.addItem("", name)
            c.currentIndexChanged.connect(lambda idx, i=i, c=c: gw.change(f"watch{i}", c.itemData(idx)))
            col.addWidget(c)
            self.combos.append(c)
        self.speed_caption = label("", "muted")
        col.addWidget(self.speed_caption)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 60)
        self.speed.valueChanged.connect(lambda v: gw.change("watch_speed", int(v)))
        col.addWidget(self.speed)
        self.thoughts_box = QCheckBox("")
        self.thoughts_box.toggled.connect(lambda on: gw.change("show_thoughts", on))
        col.addWidget(self.thoughts_box)
        self.thoughts_hint = label("", "hint", wrap=True)
        col.addWidget(self.thoughts_hint)
        col.addStretch(1)
        self.pause_button = button("", None, gw.toggle_pause)
        self.again_button = button("", "primary", lambda: gw.start("watch"))
        self.same_button = button("", None, gw.restart_same)
        self.menu_button = button("", None, lambda: gw.show_page("menu"))
        for b in (self.pause_button, self.again_button, self.same_button, self.menu_button):
            col.addWidget(b)
        root.addWidget(side)

    def sync(self):
        s = self.gw.settings
        for i, c in enumerate(self.combos):
            c.blockSignals(True)
            c.setCurrentIndex(c.findData(s[f"watch{i}"]))
            c.blockSignals(False)
        self.speed.blockSignals(True)
        self.speed.setValue(s["watch_speed"])
        self.speed.blockSignals(False)
        self.speed_caption.setText(self.gw.t("ai_speed_hint", n=s["watch_speed"]))

    def retranslate(self):
        t = self.gw.t
        self.heading.setText(t("watch_title"))
        self.hint.setText(t("watch_hint"))
        for c in self.combos:
            for i in range(c.count()):
                c.setItemText(i, t("agent_" + c.itemData(i)))
        self.thoughts_box.setText(t("thoughts"))
        self.thoughts_hint.setText(t("thoughts_hint_ai"))
        self.pause_button.setText(t("resume") if self.gw.paused else t("pause"))
        self.again_button.setText(t("again"))
        self.same_button.setText(t("same"))
        self.menu_button.setText(t("menu"))
        self.sync()


# --- статистика ---


def make_table(cols):
    table = QTableWidget(0, cols)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setShowGrid(False)
    return table


def fill_table(table, headers, rows):
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            item = QTableWidgetItem(str(text))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if c else
                                  Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(r, c, item)
    table.setFixedHeight(table.horizontalHeader().height() + 34 * max(1, len(rows)) + 6)


class ScrollPage(QScrollArea):
    def __init__(self, gw):
        super().__init__()
        self.gw = gw
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        body = page_widget()
        self.setWidget(body)
        self.col = QVBoxLayout(body)
        self.col.setContentsMargins(26, 22, 26, 22)
        self.col.setSpacing(14)
        top = QHBoxLayout()
        self.heading = label("", "heading")
        top.addWidget(self.heading)
        top.addStretch(1)
        self.menu_button = button("", None, lambda: gw.show_page("menu"))
        top.addWidget(self.menu_button)
        self.col.addLayout(top)


class StatsPage(ScrollPage):
    def __init__(self, gw):
        super().__init__(gw)
        self.solo_caption = label("", "section")
        self.col.addWidget(self.solo_caption)
        self.solo = make_table(4)
        self.col.addWidget(self.solo)
        self.race_caption = label("", "section")
        self.col.addWidget(self.race_caption)
        self.race = make_table(3)
        self.col.addWidget(self.race)
        self.empty = label("", "muted")
        self.col.addWidget(self.empty)
        self.col.addStretch(1)

    def refresh(self):
        t = self.gw.t
        self.heading.setText(t("stats_title"))
        self.menu_button.setText(t("menu"))
        self.solo_caption.setText(t("solo_records"))
        self.race_caption.setText(t("race_results"))
        rows = []
        hist = [h for h in self.gw.history() if h.get("mode") == "solo"]
        for size in SIZES:
            mine = [h for h in hist if h.get("size") == size]
            if not mine:
                continue
            best_t = min(h["seconds"] for h in mine)
            best_r = min(h["steps"] / max(h["shortest"], 1) for h in mine)
            rows.append((t("size_" + size), len(mine), fmt_time(t, best_t), f"{best_r:.2f}"))
        fill_table(self.solo, [t("col_maze"), t("col_games"), t("col_best_time"), t("col_best_ratio")], rows)
        race_rows = []
        for lv in LEVELS:
            e = self.gw.ctx.stats.entry(f"race/{lv}")
            if e["games"]:
                race_rows.append((t(lv), e["wins"], e["losses"]))
        fill_table(self.race, [t("col_level"), t("col_wins"), t("col_losses")], race_rows)
        self.empty.setText(t("no_games") if not rows and not race_rows else "")
        self.empty.setVisible(not rows and not race_rows)


# --- обучение ---


class TrainingPage(ScrollPage):
    def __init__(self, gw):
        super().__init__(gw)
        self.curve_caption = label("", "muted", wrap=True)
        self.col.addWidget(self.curve_caption)
        box = card()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(14, 14, 14, 14)
        self.chart = CurveChart()
        lay.addWidget(self.chart)
        self.col.addWidget(box)
        self.passport_caption = label("", "section")
        self.col.addWidget(self.passport_caption)
        self.passport_note = label("", "hint", wrap=True)
        self.col.addWidget(self.passport_note)
        self.table = make_table(5)
        self.col.addWidget(self.table)
        self.train_note = label("", "hint", wrap=True)
        self.col.addWidget(self.train_note)
        self.col.addStretch(1)

    def refresh(self):
        t = self.gw.t
        self.heading.setText(t("training_title"))
        self.menu_button.setText(t("menu"))
        self.curve_caption.setText(t("curve_caption"))
        self.passport_caption.setText(t("passport"))
        training = load_json(os.path.join(MODEL_DIR, "training.json"))
        passport = load_passport()
        if training and training.get("curve"):
            key = training.get("curve_key", "ratio")
            pts = [(c["episodes"], c[key]) for c in training["curve"]]
            refs = [(t("agent_" + k), v, col) for k, v, col in
                    ((k, training.get("baselines", {}).get(k), col) for k, col in
                     (("right_hand", "muted"), ("tremaux", "accent2"), ("optimist", "select")))
                    if v is not None]
            self.chart.set_data(pts, refs, t("curve_x"), t("curve_net"), self.gw.ctx.theme)
            self.chart.setVisible(True)
        else:
            self.chart.setVisible(False)
        if passport:
            agents = passport["agents"]
            headers = [t("passport_size")] + [t("agent_" + a) for a in agents]
            rows = []
            for check in passport["checks"]:
                row = [f"{check['size']}×{check['size']}" + (f" · {t('kind_' + check['kind'])}"
                                                               if check.get("kind") != "train" else "")]
                for a in agents:
                    r = check["agents"][a]
                    row.append(f"{r['mean_steps']:.0f} ({r['ratio']:.2f}×)")
                rows.append(row)
            self.table.setColumnCount(len(headers))
            fill_table(self.table, headers, rows)
            self.passport_note.setText(t("passport_note", games=passport["games_per_check"]))
            tr = passport.get("training", {})
            self.train_note.setText(t("passport_train", updates=tr.get("updates", 0),
                                      episodes=number(tr.get("episodes", 0)), steps=number(tr.get("transitions", 0)),
                                      time=fmt_time(t, tr.get("seconds", 0)), threads=tr.get("threads", "?"),
                                      params=number(passport.get("network", {}).get("parameters", 0))))
            self.table.setVisible(True)
        else:
            self.passport_note.setText(t("not_trained"))
            self.train_note.setText("")
            self.table.setVisible(False)


# --- настройки ---


class SettingsPage(ScrollPage):
    def __init__(self, gw):
        super().__init__(gw)
        box = card()
        box.setMaximumWidth(760)
        self.grid = QGridLayout(box)
        self.grid.setContentsMargins(22, 18, 22, 18)
        self.grid.setHorizontalSpacing(22)
        self.grid.setVerticalSpacing(16)
        self.grid.setColumnStretch(0, 1)
        self.rows = {}
        self.theme_combo = Combo()
        for k in theme.THEMES:
            self.theme_combo.addItem("", k)
        self.theme_combo.currentIndexChanged.connect(lambda i: gw.ctx.set_theme(self.theme_combo.itemData(i)))
        self._row("theme", self.theme_combo)
        self.lang_combo = Combo()
        self.lang_combo.addItem("Русский", "ru")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.currentIndexChanged.connect(lambda i: gw.ctx.set_lang(self.lang_combo.itemData(i)))
        self._row("language", self.lang_combo)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(gw.ctx.sound.volume)
        self.volume.valueChanged.connect(gw.ctx.set_volume)
        self._row("volume", self.volume)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 30)
        self.speed.valueChanged.connect(lambda v: (gw.change("ai_speed", int(v)), self.sync()))
        self._row("ai_speed", self.speed)
        self.col.addWidget(box)
        self.note = label("", "hint")
        self.col.addWidget(self.note)
        self.col.addStretch(1)

    def _row(self, key, widget):
        r = len(self.rows)
        names = QVBoxLayout()
        names.setSpacing(2)
        title = label("", "section")
        hint = label("", "hint", wrap=True)
        names.addWidget(title)
        names.addWidget(hint)
        self.grid.addLayout(names, r, 0)
        self.grid.addWidget(widget, r, 1)
        self.rows[key] = (title, hint)

    def retranslate(self):
        t = self.gw.t
        self.heading.setText(t("settings_title"))
        self.menu_button.setText(t("menu"))
        for key, (title, hint) in self.rows.items():
            title.setText(t(key))
            hint.setText(t(key + "_hint", n=self.gw.settings["ai_speed"]))
        for i, k in enumerate(theme.THEMES):
            self.theme_combo.setItemText(i, t(k))
        self.note.setText(t("saved_note"))
        self.sync()

    def sync(self):
        g = self.gw
        for combo, value in ((self.theme_combo, g.ctx.theme), (self.lang_combo, g.ctx.lang)):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(False)
        self.speed.blockSignals(True)
        self.speed.setValue(g.settings["ai_speed"])
        self.speed.blockSignals(False)
        self.rows["ai_speed"][1].setText(g.t("ai_speed_hint", n=g.settings["ai_speed"]))


# --- сам виджет игры ---


class GameWidget(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.t = translator(ctx.lang)
        self.settings = st.Settings(SCHEMA, ctx.file("settings.json"))
        self.session = None
        self.paused = False
        self.rng = random.Random()
        self.ai_timer = QTimer(self)
        self.ai_timer.timeout.connect(self.on_ai_tick)
        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.refresh_info)
        self.stack = QStackedWidget(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.stack)
        self.menu = MenuPage(self)
        self.play = PlayPage(self)
        self.watch = WatchPage(self)
        self.stats = StatsPage(self)
        self.training = TrainingPage(self)
        self.settings_page = SettingsPage(self)
        self.pages = {"menu": self.menu, "play": self.play, "watch": self.watch, "stats": self.stats,
                      "training": self.training, "settings": self.settings_page}
        for p in self.pages.values():
            self.stack.addWidget(p)
        # клавиши - на всё окно: у кнопок и списков нет фокуса, и без этого стрелки ловились
        # бы только когда фокус внутри игры; на скрытой игре (другая в оболочке) не срабатывают
        for d, keys in KEYS.items():
            for k in keys:
                sc = QShortcut(QKeySequence(k), self)
                sc.setContext(Qt.ShortcutContext.WindowShortcut)
                sc.activated.connect(lambda d=d: self.on_key(d))
        for keys, slot in (("Escape", lambda: self.show_page("menu")), ("N", self.on_new_key),
                           ("Т", self.on_new_key)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)
        ctx.changed.connect(self.on_ctx_changed)
        self.apply_theme()
        self.retranslate()
        self.show_page("menu")

    def minimumSizeHint(self):
        return QSize(980, 660)

    # --- общее ---

    def current_page(self):
        w = self.stack.currentWidget()
        return next(k for k, p in self.pages.items() if p is w)

    def show_page(self, name):
        if name not in ("play", "watch"):
            self.stop_timers()
        self.stack.setCurrentWidget(self.pages[name])
        if name == "stats":
            self.stats.refresh()
        elif name == "training":
            self.training.refresh()
        elif name == "settings":
            self.settings_page.sync()
        elif name == "menu":
            self.menu.sync(self.settings)
        elif name == "watch":
            self.watch.sync()

    def change(self, key, value):
        if self.settings[key] == value:
            return
        self.settings.set(key, value)
        if key in ("size", "kind", "loops", "level"):
            self.menu.sync(self.settings)
        elif key == "show_thoughts":
            self.refresh()
        elif key in ("watch_speed", "ai_speed"):
            self.watch.sync()
            self._restart_ai_timer()
        elif key.startswith("watch") and self.session is not None and self.session.mode == "watch":
            self.restart_same()

    def on_ctx_changed(self, what):
        if what == "theme":
            self.apply_theme()
        elif what == "lang":
            self.t = translator(self.ctx.lang)
            self.retranslate()

    def apply_theme(self):
        c = theme.palette(self.ctx.theme)
        for b in self.play.boards + self.watch.boards:
            b.set_theme(self.ctx.theme)
        for combo in self.findChildren(Combo):
            combo.set_arrow_color(c["text"])
        if self.current_page() == "training":
            self.training.refresh()
        self.update()

    def retranslate(self):
        self.menu.retranslate()
        self.play.retranslate()
        self.watch.retranslate()
        self.settings_page.retranslate()
        page = self.current_page()
        if page == "stats":
            self.stats.refresh()
        elif page == "training":
            self.training.refresh()
        self.refresh()

    def stop_timers(self):
        self.ai_timer.stop()
        self.clock.stop()

    def shutdown(self):
        """Окно закрывается: таймеры стоп. Потоков у игры нет."""
        self.stop_timers()
        return True

    # --- игра ---

    def start(self, mode, seed=None):
        s = self.settings
        if seed is None:
            seed = self.rng.randrange(2 ** 31)
        maze = make_maze(s["size"], s["kind"], s["loops"], seed)
        agents = [s[f"watch{i}"] for i in range(4)] if mode == "watch" else ()
        self.session = Session(mode, maze, s["level"], agents, seed)
        self.paused = False
        self.stop_timers()
        if mode == "watch":
            self.stack.setCurrentWidget(self.watch)
            self.watch.sync()
            self.watch.pause_button.setText(self.t("pause"))
        else:
            self.stack.setCurrentWidget(self.play)
            self.play.boards[1].setVisible(mode == "race")
        self._restart_ai_timer()
        self.clock.start()
        self.refresh()

    def restart_same(self):
        if self.session is None:
            return
        self.start(self.session.mode, self.session.seed)

    def _restart_ai_timer(self):
        s = self.session
        if s is None or s.mode == "solo" or s.over or self.paused:
            self.ai_timer.stop()
            return
        speed = self.settings["watch_speed"] if s.mode == "watch" else self.settings["ai_speed"]
        self.ai_timer.start(max(1, round(1000 / speed)))

    def on_key(self, d):
        s = self.session
        page = self.current_page()
        if s is None or page != "play" or s.over:
            return
        was_over = s.over
        s.move(d)
        if s.over and not was_over:
            self.finish()
        self.refresh()

    def on_new_key(self):
        if self.session is not None and self.current_page() in ("play", "watch"):
            self.start(self.session.mode)

    def on_ai_tick(self):
        s = self.session
        if s is None or self.paused:
            return
        was_over = s.over
        s.tick()
        if s.over and not was_over:
            self.finish()
        self.refresh()

    def toggle_pause(self):
        self.paused = not self.paused
        self.watch.pause_button.setText(self.t("resume") if self.paused else self.t("pause"))
        self._restart_ai_timer()

    def finish(self):
        s = self.session
        self.ai_timer.stop()
        self.clock.stop()
        if s.recorded:
            return
        s.recorded = True
        if s.mode == "solo":
            self.ctx.sound.play("win")
            self.add_history(s)
        elif s.mode == "race":
            outcome = s.outcome()
            self.ctx.stats.record(s.stats_key(), outcome)
            self.ctx.sound.play(outcome)
            self.add_history(s)
        else:
            self.ctx.sound.play("draw")

    def history(self):
        data = load_json(self.ctx.file("history.json"))
        return data if isinstance(data, list) else []

    def add_history(self, s):
        """Каждый проход человека: для статистики и для будущего генератора «посложнее»."""
        h = s.human.memory
        entry = {"mode": s.mode, "size": self.settings["size"], "kind": s.maze.kind, "loops": self.settings["loops"],
                 "seed": s.seed, "steps": h.steps, "shortest": s.maze.shortest(),
                 "seconds": round(s.elapsed(), 2), "done": s.human.done,
                 "revisits": int((h.visits > 1).sum())}
        if s.mode == "race":
            entry.update(level=s.level, result=s.outcome())
        hist = (self.history() + [entry])[-HISTORY_MAX:]
        save_json(self.ctx.file("history.json"), hist)

    # --- вид ---

    def thoughts_for(self, runner, human=False):
        if not self.settings["show_thoughts"]:
            return None
        th = {"probs": None, "explored": None}
        agent = runner.agent
        if human:
            net = load_model(STRONG)
            if net is not None and not runner.done:
                th["probs"] = NetAgent(net).think(runner.memory)[0]
        elif isinstance(agent, NetAgent) and not runner.stopped:
            th["probs"] = agent.think(runner.memory)[0]
        elif agent is not None and getattr(agent, "cheater", False):
            th["explored"] = agent.explored
        return th

    def refresh(self):
        s = self.session
        if s is None:
            return
        t = self.t
        if s.mode == "watch":
            self._refresh_watch()
            return
        pp = self.play
        human = s.human
        pp.boards[0].show_runner(human, reveal=s.over, thoughts=self.thoughts_for(human, human=True),
                                 title=t("you") if s.mode == "race" else "",
                                 subtitle=t("agent_steps", n=human.memory.steps) if s.mode == "race" else "")
        if s.mode == "race":
            ai = s.runners[1]
            pp.boards[1].show_runner(ai, reveal=s.over, thoughts=self.thoughts_for(ai),
                                     title=f"{t('ai')} · {t(s.level)}", subtitle=t("agent_steps", n=ai.memory.steps),
                                     highlight=True)
        pp.heading.setText(f"{t('app_title')} · {t('size_' + self.settings['size'])} · {t('kind_' + s.maze.kind)}")
        short = s.maze.shortest()
        if s.mode == "solo":
            if s.over:
                restyle(pp.status, "statusWin")
                pp.status.setText(t("solo_done"))
                pp.substatus.setText(t("solo_result", steps=human.memory.steps, t=fmt_time(t, s.elapsed()),
                                       short=short, extra=human.memory.steps - short))
            else:
                restyle(pp.status, "status")
                pp.status.setText(t("solo_status"))
                pp.substatus.setText(t("solo_hint"))
        else:
            if s.over:
                win = s.outcome() == "win"
                restyle(pp.status, "statusWin" if win else "statusLoss")
                pp.status.setText(t("race_win") if win else t("race_loss"))
                pp.substatus.setText(t("race_result", you=human.memory.steps, ai=s.runners[1].memory.steps,
                                       short=short))
            else:
                restyle(pp.status, "status")
                pp.status.setText(t("race_status"))
                pp.substatus.setText(t("race_hint", level=t(s.level).lower()))
        show = self.settings["show_thoughts"]
        pp.thoughts_box.blockSignals(True)
        pp.thoughts_box.setChecked(show)
        pp.thoughts_box.blockSignals(False)
        pp.thoughts_hint.setText(t("thoughts_hint_solo") if s.mode == "solo" else t("thoughts_hint_ai"))
        pp.thoughts_hint.setVisible(show)
        self.refresh_info()

    def refresh_info(self):
        s = self.session
        if s is None or s.mode == "watch":
            return
        t = self.t
        lines = [t("steps", n=s.human.memory.steps), t("time", t=fmt_time(t, s.elapsed()))]
        if s.over:
            lines.append(t("shortest", n=s.maze.shortest()))
        self.play.info.setText("\n".join(lines))

    def _refresh_watch(self):
        s = self.session
        t = self.t
        wp = self.watch
        for board, runner in zip(wp.boards, s.runners):
            if runner.missing:
                board.show_runner(None, title=t("agent_" + runner.name), subtitle=t("agent_missing"))
                continue
            m = runner.memory
            if runner.done:
                sub = t("agent_done", n=m.steps, place=runner.finish_order)
            elif runner.failed:
                sub = t("agent_failed", n=m.steps)
            else:
                sub = t("agent_steps", n=m.steps)
            if getattr(runner.agent, "cheater", False):
                sub += " · " + t("cheater_note", n=len(runner.agent.explored))
            board.show_runner(runner, reveal=getattr(runner.agent, "cheater", False) or s.over,
                              thoughts=self.thoughts_for(runner), title=t("agent_" + runner.name), subtitle=sub,
                              highlight=runner.finish_order == 1)
        show = self.settings["show_thoughts"]
        wp.thoughts_box.blockSignals(True)
        wp.thoughts_box.setChecked(show)
        wp.thoughts_box.blockSignals(False)
        wp.thoughts_hint.setVisible(show)

