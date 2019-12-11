#!/usr/bin/env bash
#
# Run (Python) script with correct PYTHONPATH.
#

export PYTHONPATH="$(dirname "$(realpath "$0")")/src:$PYTHONPATH"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore

if [ "$1" == "" ]; then
    echo -e "Execute Python file from src directory with correct PYTHONPATH\n" >&1
    echo -e "    Usage: $(basename $0) PYTHON_FILE\n" >&2
    exit 1
fi

if [ -x "$1" ]; then
    exec "$@"
else
    exec python3 "$@"
fi
