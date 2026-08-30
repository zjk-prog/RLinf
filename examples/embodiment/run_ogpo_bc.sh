#!/bin/bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
CONFIG_NAME="${1:-maniskill_ogpo_bc}"
if [ "$#" -gt 0 ]; then
    shift
fi

export EMBODIED_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

if [ -z "${MS_ASSET_DIR:-}" ]; then
    echo "MS_ASSET_DIR must point to the ManiSkill asset directory." >&2
    echo "Example: export MS_ASSET_DIR=/data/maniskill" >&2
    exit 1
fi

# Prepare the dataset once before training:
# python -m mani_skill.utils.download_demo "PickCube-v1"
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path "$MS_ASSET_DIR/demos/PickCube-v1/motionplanning/trajectory.h5" \
#   --use-first-env-state -c pd_ee_delta_pos -o state --save-traj \
#   --num-envs 1 -b physx_cpu
# SDE and ODE eval MP4s are written under <log_dir>/video/bc_eval/{sde,ode}.

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "$LOG_DIR"

python -u "${EMBODIED_PATH}/train_ogpo_bc.py" \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name "$CONFIG_NAME" \
    runner.logger.log_path="$LOG_DIR" \
    "$@" 2>&1 | tee "$LOG_DIR/run_ogpo_bc.log"
