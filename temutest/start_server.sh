#!/bin/bash
set -euo pipefail

# ============================================================================
#  启动 TRT-LLM + FlexKV 服务
#  Workload: gemma3-4b, ISL=160, OSL=1, single H20 GPU
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLEXKV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================== Configurable Parameters ========================

MODEL_PATH="${MODEL_PATH:-/raid/model/gemma-3-4b-novision}"
SERVE_PORT="${SERVE_PORT:-6000}"

# TRT-LLM serving params (single H20 GPU)
TP_SIZE="${TP_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-256}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-65536}"

# FlexKV cache params
CPU_CACHE_GB="${CPU_CACHE_GB:-64}"
SSD_CACHE_GB="${SSD_CACHE_GB:-1024}"
SSD_CACHE_DIR="${SSD_CACHE_DIR:-/raid/wenpengw/code/FlexKV/flexkv_ssd/}"

# ======================== Helper Functions ========================

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }

# ======================== Pre-flight Checks ========================

log "============================================================"
log " FlexKV + TRT-LLM Server"
log "============================================================"
log " Model:        ${MODEL_PATH}"
log " TP:           ${TP_SIZE}"
log " GPU:          single H20"
log " CPU Cache:    ${CPU_CACHE_GB} GB"
log " SSD Cache:    ${SSD_CACHE_GB} GB"
log " Serve Port:   ${SERVE_PORT}"
log "============================================================"

[ -d "$MODEL_PATH" ]  || die "Model path not found: ${MODEL_PATH}"
command -v trtllm-serve > /dev/null || die "trtllm-serve not found. Install TRT-LLM first."
python3 -c "import flexkv" 2>/dev/null || die "flexkv not importable. Run: cd ${FLEXKV_DIR} && ./build.sh"

# ======================== Generate Config Files ========================

log "[1/3] Generating config files..."

CONFIG_DIR="${SCRIPT_DIR}/config"
mkdir -p "${CONFIG_DIR}"

cat > "${CONFIG_DIR}/flexkv_config.yml" <<EOF
cpu_cache_gb: ${CPU_CACHE_GB}
ssd_cache_gb: ${SSD_CACHE_GB}
ssd_cache_dir: ${SSD_CACHE_DIR}
enable_gds: false
EOF

cat > "${CONFIG_DIR}/extra-llm-api-config.yml" <<EOF
enable_chunked_prefill: false
kv_cache_config:
  enable_partial_reuse: false
  free_gpu_memory_fraction: 0.90
# kv_connector_config:
#   connector_module: "flexkv.integration.tensorrt_llm.trtllm_adapter"
#   connector_scheduler_class: "FlexKVSchedulerConnector"
#   connector_worker_class: "FlexKVWorkerConnector"
print_iter_log: false
EOF

log "  ${CONFIG_DIR}/flexkv_config.yml"
log "  ${CONFIG_DIR}/extra-llm-api-config.yml"

# ======================== Set Environment Variables ========================

log "[2/3] Setting environment variables..."

export TENSORRT_LLM_USE_FLEXKV=0
export FLEXKV_CONFIG_PATH="${CONFIG_DIR}/flexkv_config.yml"
export FLEXKV_LOG_LEVEL="${FLEXKV_LOG_LEVEL:-INFO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# monitor metrics
export FLEXKV_ENABLE_METRICS=0
export FLEXKV_PY_METRICS_PORT=8080

log "  TENSORRT_LLM_USE_FLEXKV=${TENSORRT_LLM_USE_FLEXKV}"
log "  FLEXKV_CONFIG_PATH=${FLEXKV_CONFIG_PATH}"
log "  FLEXKV_LOG_LEVEL=${FLEXKV_LOG_LEVEL}"
log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "  FLEXKV_ENABLE_METRICS=${FLEXKV_ENABLE_METRICS}"
log "  FLEXKV_PY_METRICS_PORT=${FLEXKV_PY_METRICS_PORT}"

# ======================== Kill Existing Server ========================

if command -v lsof > /dev/null && lsof -i :"${SERVE_PORT}" -t > /dev/null 2>&1; then
    log "  Killing existing process on port ${SERVE_PORT}..."
    kill "$(lsof -i :"${SERVE_PORT}" -t)" 2>/dev/null || true
    sleep 3
fi

# ======================== Start Server (foreground) ========================

log "[3/3] Starting TRT-LLM + FlexKV server (foreground)..."
log "  Press Ctrl+C to stop."
log ""

trtllm-serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${SERVE_PORT}" \
    --backend pytorch \
    --tp_size "${TP_SIZE}" \
    --ep_size "${TP_SIZE}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_num_tokens "${MAX_NUM_TOKENS}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --extra_llm_api_options "${CONFIG_DIR}/extra-llm-api-config.yml" 2>&1 | tee "${SCRIPT_DIR}/server.log"
