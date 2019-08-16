#!/usr/bin/env bash
#
# Run (Python) script with correct PYTHONPATH.
#

PYTHONPATH="$(dirname "$(realpath "$0")"):$PYTHONPATH"
export PYTHONPATH

if [ -x "$1" ]; then
    exec "$@"
else
    exec python3 "$@"
fi
