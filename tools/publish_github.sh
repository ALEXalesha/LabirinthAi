#!/usr/bin/env bash
# Публикация на GitHub (адрес - REPO в tools/publish.conf):
#
#   bash tools/publish_github.sh            # только код
#   bash tools/publish_github.sh v1.0.0     # код и тег
#   DRY_RUN=1 bash tools/publish_github.sh  # собрать и проверить, ничего не отправлять
#
# Тонкая обёртка: вся работа - в общем скрипте PythonAiGames/tools/publish_game.sh (папка
# рядом с этой игрой), он чистит личное по всей истории и проверяет его до пуша. Особенности
# игры (что вырезать, какие слова запрещены) - в tools/publish.conf. Если игра лежит не
# рядом с PythonAiGames (копия в wip), путь к общему скрипту - в переменной PUBLISH_GAME.
set -euo pipefail
GAME="$(cd "$(dirname "$0")/.." && pwd)"
COMMON="${PUBLISH_GAME:-$GAME/../PythonAiGames/tools/publish_game.sh}"
[ -f "$COMMON" ] || { echo "нет общего скрипта $COMMON - задайте PUBLISH_GAME" >&2; exit 1; }
exec bash "$COMMON" "$GAME" "$@"
