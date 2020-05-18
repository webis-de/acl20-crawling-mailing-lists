#!/usr/bin/env bash
#
# Run (Python) script with correct PYTHONPATH.
#

SRC_PATH="$(dirname "$(realpath "$0")")/src"
TMP_PATH="$(realpath "${SRC_PATH}/../data/tmp")"

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
elif [[ "$1" =~ ^spark ]]; then
    cmd="$1"
    mkdir -p "$TMP_PATH"
    shift
    pushd "${SRC_PATH}" && zip -r "${TMP_PATH}/py-files.zip" ./* && popd || exit 1

    case "$cmd" in
        spark-fat)
            exec spark-submit --master yarn --py-files "${TMP_PATH}/py-files.zip" \
                --num-executors 200 --executor-cores 6 --executor-memory 20G --conf spark.task.cpus=6 \
                --conf spark.yarn.executor.memoryOverhead=8192 "$@"
            ;;
        *)
            exec spark-submit --master yarn --py-files "${TMP_PATH}/py-files.zip" \
                --num-executors 300 --executor-memory 2G "$@"
            ;;
    esac
elif [ -x "$1" ]; then
    exec "$@"
else
    exec python3 "$@"
fi
