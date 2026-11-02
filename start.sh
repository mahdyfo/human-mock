#!/bin/sh
# Starts humanmock. The first time, it sets itself up in a .venv folder here.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/humanmock ]; then
  PYTHON=""
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON" ]; then
    echo "humanmock needs Python 3.11 or newer. Get it from https://www.python.org/downloads/"
    exit 1
  fi
  echo "Setting up humanmock. This happens once and takes a few minutes."
  "$PYTHON" -m venv .venv
  .venv/bin/python -m pip install --quiet --disable-pip-version-check -e .
fi

exec .venv/bin/humanmock start "$@"
