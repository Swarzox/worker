#!/bin/bash
# Z-Image Turbo - Entrypoint pour Vast.ai Serverless

set -e

mkdir -p /var/log/zimage
: > /var/log/zimage/model.log
: > /var/log/zimage/ready.log

# torch.compile (Inductor/Triton) links against libcuda.so (unversioned),
# but many GPU hosts only provide libcuda.so.1. Create a local symlink if needed.
ensure_libcuda() {
    if [ -e "/usr/lib/x86_64-linux-gnu/libcuda.so" ] || [ -e "/lib/x86_64-linux-gnu/libcuda.so" ] || [ -e "/usr/local/nvidia/lib64/libcuda.so" ] || [ -e "/usr/local/nvidia/lib/libcuda.so" ]; then
        return 0
    fi

    for candidate in \
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1" \
        "/lib/x86_64-linux-gnu/libcuda.so.1" \
        "/usr/local/nvidia/lib64/libcuda.so.1" \
        "/usr/local/nvidia/lib/libcuda.so.1"
    do
        if [ -e "$candidate" ]; then
            dir="$(dirname "$candidate")"
            ln -sf "$(basename "$candidate")" "$dir/libcuda.so" 2>/dev/null || true
            export LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH:-}"
            export LIBRARY_PATH="$dir:${LIBRARY_PATH:-}"
            echo "[entrypoint] libcuda.so missing; created symlink in $dir"
            return 0
        fi
    done

    echo "[entrypoint] WARNING: libcuda.so.1 not found; torch.compile may fail." >&2
    return 0
}

# === CONFIGURER LES VARIABLES VAST.AI ===
export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
export WORKER_PORT="${WORKER_PORT:-5000}"
export RUN_PYWORKER="${RUN_PYWORKER:-1}"
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-flash}"
export ZIMAGE_COMPILE="${ZIMAGE_COMPILE:-1}"

echo "[entrypoint] === Configuration ==="
echo "REPORT_ADDR=$REPORT_ADDR"
echo "RUN_PYWORKER=$RUN_PYWORKER"
echo "DIFFUSERS_ATTN_BACKEND=$DIFFUSERS_ATTN_BACKEND"
echo "ZIMAGE_COMPILE=$ZIMAGE_COMPILE"
echo "CONTAINER_ID=${CONTAINER_ID:-NOT_SET}"
echo "[entrypoint] ======================"

SERVER_PID=""
WORKER_PID=""

# Fonction pour nettoyer les processus
cleanup() {
    echo "[entrypoint] Cleaning up..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
        kill $WORKER_PID 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT
ensure_libcuda

if [ "$#" -gt 0 ]; then
    echo "[entrypoint] Running custom command: $*"
    exec "$@"
fi

echo "[entrypoint] Starting Z-Image Turbo FastAPI server in background..."
python /app/server.py &
SERVER_PID=$!

if [ "$RUN_PYWORKER" = "1" ]; then
    echo "[entrypoint] Starting built-in PyWorker in background..."
    python /app/worker.py &
    WORKER_PID=$!

    # Attendre que l'un des processus se termine
    echo "[entrypoint] Server + Worker running. Waiting..."
    wait -n "$SERVER_PID" "$WORKER_PID" 2>/dev/null || true

    echo "[entrypoint] A process exited. Checking which one..."
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[entrypoint] Server died! Exiting..."
        exit 1
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "[entrypoint] Worker died! Exiting..."
        exit 1
    fi
else
    echo "[entrypoint] RUN_PYWORKER=0 - running server only"
    wait "$SERVER_PID"
fi
