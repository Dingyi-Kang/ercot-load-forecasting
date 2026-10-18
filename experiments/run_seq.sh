#!/bin/bash
# sequential experiment runner (no pgrep-based waiting)
cd "$(dirname "$0")/.."
.venv/bin/python -m src.elf.backtest "$@"
