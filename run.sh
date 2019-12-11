#!/usr/bin/env bash
#
# Run (Python) script with correct PYTHONPATH.
#

SRC_PATH="$(dirname "$(realpath "$0")")/src"

export PYTHONPATH="${SRC_PATH}:$PYTHONPATH"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONWARNINGS=ignore

if [ "$1" == "" ]; then
    echo -e "Execute Python file from src directory with correct PYTHONPATH\n" >&1
    echo -e "    Usage: $(basename $0) PYTHON_FILE\n" >&2
    exit 1
fi

if [ "$1" == "explorer" ]; then
    shift
    export FLASK_APP="${SRC_PATH}/explorer/explorer.py"
    flask run "$@"
elif [ -x "$1" ]; then
    exec "$@"
else
    exec python3 "$@"
fi
