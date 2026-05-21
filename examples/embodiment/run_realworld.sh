#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

# Keep Ray temp/session artifacts off a nearly-full root /tmp by default.
RAY_USER="${USER:-$(id -un)}"
if [ -z "${RAY_TMPDIR:-}" ]; then
    if [ -d "/data/${RAY_USER}" ] && [ -w "/data/${RAY_USER}" ]; then
        export RAY_TMPDIR="/data/${RAY_USER}/ray_tmp"
    elif [ -d "/data" ] && [ -w "/data" ]; then
        export RAY_TMPDIR="/data/ray_tmp_${RAY_USER}"
    else
        export RAY_TMPDIR="/dev/shm/ray_tmp_${RAY_USER}"
    fi
fi
mkdir -p "${RAY_TMPDIR}"

if [ -z "$1" ]; then
    CONFIG_NAME="realworld_sac_cnn"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
echo "Using RAY_TMPDIR=${RAY_TMPDIR}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}