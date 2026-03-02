#!/bin/bash
set -euo pipefail

# ============================================================================
#  启动 aiperf 压测
#  前置条件: start_server.sh 已启动且服务已就绪
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ======================== Configurable Parameters ========================

MODEL_PATH="${MODEL_PATH:-/raid/model/gemma-3-4b-novision}"
SERVE_PORT="${SERVE_PORT:-6000}"
TRACE_FILE="${TRACE_FILE:-${SCRIPT_DIR}/trace_hashids_gpu1_qps120_isl160_osl1_0.5h.jsonl}"

# aiperf goodput SLOs (平响100ms, p999=160ms)
GOODPUT_SLOS="${GOODPUT_SLOS:-time_to_first_token:160}"

# Output
RESULT_DIR="${SCRIPT_DIR}/results/${TIMESTAMP}"

# ======================== Helper Functions ========================

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }

# ======================== Pre-flight Checks ========================

log "============================================================"
log " aiperf Benchmark"
log "============================================================"
log " Server:       http://localhost:${SERVE_PORT}"
log " Trace:        ${TRACE_FILE}"
log " Results:      ${RESULT_DIR}"
log "============================================================"

[ -f "$TRACE_FILE" ] || die "Trace file not found: ${TRACE_FILE}"
command -v aiperf > /dev/null || die "aiperf not found. Run: pip install aiperf"

# ======================== Wait for Server Ready ========================

log "[1/4] Checking server readiness..."

MAX_WAIT=300
INTERVAL=5
ELAPSED=0

while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    if curl -sf "http://localhost:${SERVE_PORT}/v1/models" > /dev/null 2>&1; then
        log "  Server is READY."
        break
    fi
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
    log "  Waiting for server... (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    die "Server not reachable at localhost:${SERVE_PORT} within ${MAX_WAIT}s. Is start_server.sh running?"
fi

# ======================== Sanity Check ========================

log "[2/4] Sending sanity check request..."

SANITY_RESP=$(curl -sf "http://localhost:${SERVE_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$(basename "${MODEL_PATH}")\",\"prompt\":\"Hello\",\"max_tokens\":1}" 2>&1) || true

if echo "$SANITY_RESP" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    log "  Sanity check PASSED."
else
    log "  WARNING: Sanity check got unexpected response (server may still work)."
    log "  Response: ${SANITY_RESP:0:200}"
fi

# ======================== TCP Tuning (avoid port exhaustion) ========================

log "[3/3] Tuning TCP for high-QPS localhost benchmark..."

# server + client share the same port pool on localhost; widen it
sysctl -w net.ipv4.ip_local_port_range="1024 65535"  > /dev/null 2>&1 || true
sysctl -w net.ipv4.tcp_fin_timeout=10                 > /dev/null 2>&1 || true
sysctl -w net.ipv4.tcp_tw_reuse=1                     > /dev/null 2>&1 || true
sysctl -w net.core.somaxconn=65535                     > /dev/null 2>&1 || true

log "  ip_local_port_range = $(cat /proc/sys/net/ipv4/ip_local_port_range)"
log "  tcp_fin_timeout     = $(cat /proc/sys/net/ipv4/tcp_fin_timeout)"
log "  tcp_tw_reuse        = $(cat /proc/sys/net/ipv4/tcp_tw_reuse)"

# ======================== Run aiperf ========================

log "[4/4] Running aiperf benchmark..."
mkdir -p "$RESULT_DIR"

aiperf profile \
    --model "$(basename "${MODEL_PATH}")" \
    --tokenizer "${MODEL_PATH}" \
    --endpoint-type completions \
    --streaming \
    --url "http://localhost:${SERVE_PORT}" \
    --input-file "${TRACE_FILE}" \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule \
    --isl-block-size 150 \
    --goodput "${GOODPUT_SLOS}" \
    --artifact-dir "${RESULT_DIR}" \
    --export-level raw \
    --warmup-request-count 10 \
    --ui none \
    --use-legacy-max-tokens \
    --extra-inputs "max_tokens:1" \
    2>&1 | tee "${RESULT_DIR}/aiperf.log"

# ======================== Summary ========================

log ""
log "============================================================"
log " Benchmark Complete!"
log "============================================================"
log " Results:     ${RESULT_DIR}/"
log " aiperf log:  ${RESULT_DIR}/aiperf.log"
log ""
log " Key artifacts:"
ls -lh "${RESULT_DIR}"/profile_export* 2>/dev/null | while read -r line; do
    log "   $line"
done
log "============================================================"
