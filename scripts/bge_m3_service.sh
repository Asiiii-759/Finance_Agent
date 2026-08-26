#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${BGE_M3_RUNTIME_DIR:-${PROJECT_DIR}/.runtime/bge-m3}"
MODEL_DIR="${BGE_M3_MODEL_DIR:-${PROJECT_DIR}/.runtime/models/bge-m3}"
VLLM_BIN="${BGE_M3_VLLM_BIN:-${PROJECT_DIR}/.runtime/vllm-venv/bin/vllm}"
HOST="${BGE_M3_HOST:-127.0.0.1}"
PORT="${BGE_M3_PORT:-8001}"
PID_FILE="${RUNTIME_DIR}/server.pid"
LOG_FILE="${RUNTIME_DIR}/server.log"

case "${1:-}" in
  start)
    [[ -x "${VLLM_BIN}" ]] || { echo "vLLM executable not found: ${VLLM_BIN}" >&2; exit 1; }
    [[ -f "${MODEL_DIR}/config.json" ]] || { echo "BGE-M3 model not found: ${MODEL_DIR}" >&2; exit 1; }
    mkdir -p "${RUNTIME_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
      echo "BGE-M3 is already running (PID $(<"${PID_FILE}"))"
      exit 0
    fi
    nohup "${VLLM_BIN}" serve "${MODEL_DIR}" \
      --served-model-name BAAI/bge-m3 \
      --runner pooling \
      --dtype half \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.65 \
      --host "${HOST}" \
      --port "${PORT}" >"${LOG_FILE}" 2>&1 &
    echo "$!" >"${PID_FILE}"
    echo "BGE-M3 starting on http://${HOST}:${PORT} (PID $!, log ${LOG_FILE})"
    ;;
  stop)
    [[ -f "${PID_FILE}" ]] || { echo "BGE-M3 is not running"; exit 0; }
    PID="$(<"${PID_FILE}")"
    kill "${PID}"
    for _ in {1..30}; do
      kill -0 "${PID}" 2>/dev/null || break
      sleep 1
    done
    kill -0 "${PID}" 2>/dev/null && { echo "BGE-M3 did not stop within 30 seconds" >&2; exit 1; }
    rm -f "${PID_FILE}"
    echo "BGE-M3 stopped"
    ;;
  status)
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
      echo "BGE-M3 is running (PID $(<"${PID_FILE}"))"
      exit 0
    fi
    echo "BGE-M3 is not running"
    exit 1
    ;;
  *)
    echo "Usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
