#!/usr/bin/env bash
#
# Run (Python) script with correct PYTHONPATH.
#

export PYTHONPATH="$(dirname "$(realpath "$0")"):$PYTHONPATH"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore

if [ -x "$1" ]; then
    exec "$@"
else
    exec python3 "$@"
fi
