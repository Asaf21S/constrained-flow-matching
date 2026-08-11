#!/bin/bash
# Submits one training job per config so they run concurrently.
#
# Runs on the login node -- it only calls sbatch, it does not compute anything.
#
#   scripts/run_sweep.sh constrained_fm/configs/*.yaml
#   SMOKE=1 POOL_TIME=00:15:00 TRAIN_TIME=00:40:00 scripts/run_sweep.sh constrained_fm/configs/baseline.yaml
#
# Configs sharing a conditioning pool must not build it concurrently, so a missing pool
# gets exactly one build job and every training job that needs it is chained behind that
# job with --dependency=afterok:<jobid>. Jobs with a cached pool start immediately.

set -eo pipefail
cd "$(dirname "$0")/.."

[ $# -eq 0 ] && { echo "usage: run_sweep.sh <config.yaml> [config.yaml ...]"; exit 1; }

SMOKE_FLAG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

POOL_OPTS=""
TRAIN_OPTS=""
[ -n "${POOL_TIME:-}" ] && POOL_OPTS="--time=${POOL_TIME}"
[ -n "${TRAIN_TIME:-}" ] && TRAIN_OPTS="--time=${TRAIN_TIME}"

declare -A POOL_JOB

for CONFIG in "$@"; do
    # Fails loudly on a typo before anything reaches the queue.
    python3 -m constrained_fm.scripts.check_config "$CONFIG" $SMOKE_FLAG > /dev/null

    RUN_ID=$(python3 -m constrained_fm.scripts.check_config --run-id "$CONFIG" $SMOKE_FLAG)
    POOL_KEY=$(python3 -m constrained_fm.scripts.check_config --pool-key "$CONFIG" $SMOKE_FLAG)
    POOL_FILE=$(python3 -m constrained_fm.scripts.check_config --pool-path "$CONFIG" $SMOKE_FLAG)

    DEP=""
    if [ ! -f "$POOL_FILE" ]; then
        if [ -z "${POOL_JOB[$POOL_KEY]:-}" ]; then
            POOL_JOB[$POOL_KEY]=$(sbatch --parsable $POOL_OPTS scripts/run_pool.sh "$CONFIG" $SMOKE_FLAG)
            echo "pool   $(basename "$POOL_FILE")  -> job ${POOL_JOB[$POOL_KEY]}"
        fi
        DEP="--dependency=afterok:${POOL_JOB[$POOL_KEY]}"
    fi

    TRAIN_JOB=$(sbatch --parsable $DEP $TRAIN_OPTS scripts/run_train.sh "$CONFIG" $SMOKE_FLAG)
    echo "train  ${RUN_ID}  -> job ${TRAIN_JOB} ${DEP:+(waits for pool)}"
done

echo
echo "queue:"
squeue -u "$USER" -o "%.10i %.12j %.10T %.11M %.20R %E"
