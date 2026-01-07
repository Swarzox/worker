#!/bin/bash
# Z-Image Turbo - Deploiement Vast.ai (simple)
#
# Objectif: garder le flow minimal
# - build-base : image base (deps + modele dans /models)
# - build      : image app (code only) basee sur l'image base
# - up/down    : creation/suppression endpoint serverless
# - smoke      : test /health + /generate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

# Load .env if present
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

# Docker images
DOCKER_USER="${DOCKER_USER:-swarzox}"
DOCKER_LOGIN_USER="${DOCKER_LOGIN_USER:-}"
DOCKER_LOGIN_PASS="${DOCKER_LOGIN_PASS:-}"
IMAGE_NAME="${IMAGE_NAME:-z-image-turbo}"
BASE_IMAGE_NAME="${BASE_IMAGE_NAME:-${IMAGE_NAME}-base}"

TAG="${TAG:-latest}"
BASE_TAG="${BASE_TAG:-${TAG}}"

# Vast serverless
ENDPOINT_NAME="${ENDPOINT_NAME:-z-image-turbo}"
GPU_TYPE="${GPU_TYPE:-RTX_4090}"
DISK_SIZE="${DISK_SIZE:-150}"
MIN_WORKERS="${MIN_WORKERS:-1}"
MAX_WORKERS="${MAX_WORKERS:-5}"

# Filters (avoid slow 4090 hosts)
MIN_DRIVER_VERSION="${MIN_DRIVER_VERSION:-570.86.15}"
MIN_PCIE_BW="${MIN_PCIE_BW:-20}"

# Runtime options
ZIMAGE_COMPILE="${ZIMAGE_COMPILE:-1}"
DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-flash}"
ZIMAGE_MAX_QUEUE="${ZIMAGE_MAX_QUEUE:-20}"
WORKER_LAUNCH_ARGS="${WORKER_LAUNCH_ARGS:-}"

# Build options
HF_TOKEN="${HF_TOKEN:-}"
CACHE_DIT_REF="${CACHE_DIT_REF:-main}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODEL_BASE_IMAGE="$DOCKER_USER/$BASE_IMAGE_NAME:$BASE_TAG"
FULL_IMAGE="$DOCKER_USER/$IMAGE_NAME:$TAG"
MODEL_SRC_IMAGE="${MODEL_SRC_IMAGE:-$MODEL_BASE_IMAGE}"

show_help() {
  cat <<EOF
Z-Image Turbo - Vast.ai Serverless (simple)

Usage:
  $0 build-base
  $0 build
  $0 deploy
  $0 up
  $0 down
  $0 status
  $0 logs
  $0 clean-docker
  $0 smoke

Env vars (.env supporte):
  DOCKER_USER=$DOCKER_USER
  DOCKER_LOGIN_USER=$DOCKER_LOGIN_USER
  DOCKER_LOGIN_PASS=*** (use password-stdin)
  TAG=$TAG
  BASE_TAG=$BASE_TAG
  ENDPOINT_NAME=$ENDPOINT_NAME
	  GPU_TYPE=$GPU_TYPE
	  DISK_SIZE=$DISK_SIZE
	  MIN_WORKERS=$MIN_WORKERS
	  MAX_WORKERS=$MAX_WORKERS
	  MIN_DRIVER_VERSION=$MIN_DRIVER_VERSION
	  MIN_PCIE_BW=$MIN_PCIE_BW

Perf inference:
  ZIMAGE_COMPILE=$ZIMAGE_COMPILE
  DIFFUSERS_ATTN_BACKEND=$DIFFUSERS_ATTN_BACKEND
  ZIMAGE_MAX_QUEUE=$ZIMAGE_MAX_QUEUE
EOF
}

docker_login_if_needed() {
  if [ -z "${DOCKER_LOGIN_USER:-}" ] || [ -z "${DOCKER_LOGIN_PASS:-}" ]; then
    return 0
  fi
  echo "Logging into Docker Hub as ${DOCKER_LOGIN_USER}..."
  echo "${DOCKER_LOGIN_PASS}" | docker login --username "${DOCKER_LOGIN_USER}" --password-stdin >/dev/null
}

check_hf_token() {
  if [ "${SKIP_MODEL_DOWNLOAD}" = "1" ]; then
    return 0
  fi
  if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "${RED}Erreur: HF_TOKEN non configure (requis pour build-base)${NC}"
    echo "  Fix: export HF_TOKEN=hf_..."
    exit 1
  fi
}

check_requirements() {
  local has_error=0

  if ! command -v vastai &>/dev/null; then
    echo -e "${RED}Erreur: vastai CLI non installe${NC}"
    echo "  Fix: pip install vastai"
    has_error=1
  fi
  if ! command -v jq &>/dev/null; then
    echo -e "${RED}Erreur: jq non installe${NC}"
    echo "  Fix: brew install jq"
    has_error=1
  fi
  if ! command -v docker &>/dev/null; then
    echo -e "${RED}Erreur: docker non installe${NC}"
    echo "  Fix: https://docs.docker.com/get-docker/"
    has_error=1
  fi

  if [ "$has_error" -eq 1 ]; then
    exit 1
  fi

  if ! vastai show user &>/dev/null 2>&1; then
    echo -e "${RED}Erreur: Non connecte a Vast.ai${NC}"
    echo "  Fix: vastai set api-key VOTRE_API_KEY"
    exit 1
  fi
}

get_workergroup_launch_args() {
  local args=""
  args="${args}--env ZIMAGE_COMPILE=${ZIMAGE_COMPILE} "
  args="${args}--env DIFFUSERS_ATTN_BACKEND=${DIFFUSERS_ATTN_BACKEND} "
  args="${args}--env ZIMAGE_MAX_QUEUE=${ZIMAGE_MAX_QUEUE:-20} "
  if [ -n "$WORKER_LAUNCH_ARGS" ]; then
    args="${args}${WORKER_LAUNCH_ARGS} "
  fi
  echo "${args% }"
}

show_usage_examples() {
  local endpoint_data
  endpoint_data=$(vastai show endpoints --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\")" 2>/dev/null || true)
  local api_key
  api_key=$(echo "$endpoint_data" | jq -r '.api_key' 2>/dev/null || true)

  if [ -z "$api_key" ] || [ "$api_key" = "null" ]; then
    echo -e "${YELLOW}Endpoint cree, mais API key introuvable (attends quelques secondes puis relance: $0 status)${NC}"
    return 0
  fi

  echo -e "\n${GREEN}========================================${NC}"
  echo -e "${GREEN}=== ENDPOINT PRET ===${NC}"
  echo -e "${GREEN}========================================${NC}"
  echo "Endpoint: $ENDPOINT_NAME"
  echo "API Key:  ${api_key:0:8}...  (exporte-la via VAST_ENDPOINT_API_KEY)"

  echo -e "\n${YELLOW}# Test (python):${NC}"
  echo "export VAST_ENDPOINT_API_KEY=\"<endpoint_api_key>\""
  echo "python client.py --endpoint-name \"$ENDPOINT_NAME\" --health"
  echo "python client.py --endpoint-name \"$ENDPOINT_NAME\" --prompt \"A cute robot cat\" --steps 8"
}

cmd="${1:-help}"

case "$cmd" in
  build-base)
    echo -e "${YELLOW}=== Build & Push Base Image (deps + modele) ===${NC}"
    cd "$PROJECT_DIR"
    docker_login_if_needed
    echo "Base image: $MODEL_BASE_IMAGE"
    if [ "${SKIP_MODEL_DOWNLOAD}" = "1" ]; then
      echo -e "${YELLOW}SKIP_MODEL_DOWNLOAD=1 -> copy model from existing image (no HF token)${NC}"
      echo "Model source image: $MODEL_SRC_IMAGE"
      if ! docker pull "$MODEL_SRC_IMAGE" >/dev/null; then
        echo -e "${RED}Erreur: image modele introuvable: $MODEL_SRC_IMAGE${NC}"
        echo "  Fix: set SKIP_MODEL_DOWNLOAD=0 and provide HF_TOKEN to build from scratch"
        exit 1
      fi
      DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
        --target base_copy \
        --build-arg BASE_IMAGE="pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime" \
        --build-arg MODEL_SRC_IMAGE="$MODEL_SRC_IMAGE" \
        --build-arg CACHE_DIT_REF="${CACHE_DIT_REF}" \
        --build-arg FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" \
        -f Dockerfile.base \
        -t "$MODEL_BASE_IMAGE" .
    else
      check_hf_token
      DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
        --target base_download \
        --secret id=hf_token,env=HF_TOKEN \
        --build-arg BASE_IMAGE="pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime" \
        --build-arg CACHE_DIT_REF="${CACHE_DIT_REF}" \
        --build-arg FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION}" \
        -f Dockerfile.base \
        -t "$MODEL_BASE_IMAGE" .
    fi
    docker push "$MODEL_BASE_IMAGE"
    echo -e "${GREEN}Done: $MODEL_BASE_IMAGE${NC}"
    ;;

  build|build-fast)
    echo -e "${YELLOW}=== Build & Push App Image (code-only) ===${NC}"
    cd "$PROJECT_DIR"
    docker_login_if_needed
    echo "Base image: $MODEL_BASE_IMAGE"
    echo "App image:  $FULL_IMAGE"
    echo "Pulling base image..."
    if ! docker pull "$MODEL_BASE_IMAGE"; then
      echo -e "${RED}Erreur: base image introuvable sur le registry: $MODEL_BASE_IMAGE${NC}"
      echo "  Fix: $0 build-base"
      exit 1
    fi
    docker build --platform linux/amd64 \
      --build-arg BASE_IMAGE="$MODEL_BASE_IMAGE" \
      -t "$FULL_IMAGE" .
    docker push "$FULL_IMAGE"
    echo -e "${GREEN}Done: $FULL_IMAGE${NC}"
    ;;

  deploy)
    "$0" build
    "$0" up
    ;;

  up)
    echo -e "${YELLOW}=== Deploy Serverless Endpoint ===${NC}"
    check_requirements
    echo "Endpoint: $ENDPOINT_NAME"
    echo "Image:    $FULL_IMAGE"

    echo "Nettoyage (workergroups/endpoints/templates existants)..."
    wg_ids=$(vastai show workergroups --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\") | .id" 2>/dev/null || true)
    if [ -n "${wg_ids:-}" ]; then
      for wg_id in $wg_ids; do
        vastai delete workergroup "$wg_id" 2>/dev/null || true
      done
      sleep 2
    fi

    endpoint_id=$(vastai show endpoints --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\") | .id" 2>/dev/null | head -1 || true)
    if [ -n "${endpoint_id:-}" ] && [ "$endpoint_id" != "null" ]; then
      vastai delete endpoint "$endpoint_id" 2>/dev/null || true
      sleep 2
    fi

    tmpl_ids=$(vastai search templates "name=$ENDPOINT_NAME" --raw 2>/dev/null | jq -r '.[].id' 2>/dev/null || true)
    if [ -n "${tmpl_ids:-}" ]; then
      for tmpl_id in $tmpl_ids; do
        vastai delete template "$tmpl_id" 2>/dev/null || true
      done
      sleep 1
    fi

	    echo -e "\n${YELLOW}[1/3] Creation template...${NC}"
	    template_search_params="gpu_name=$GPU_TYPE num_gpus=1 disk_space>=$DISK_SIZE reliability>=0.95"
	    if [ -n "${MIN_DRIVER_VERSION:-}" ] && [ "${MIN_DRIVER_VERSION:-0}" != "0" ]; then
	      template_search_params="${template_search_params} driver_version>=${MIN_DRIVER_VERSION}"
	    fi
	    if [ -n "${MIN_PCIE_BW:-}" ] && [ "${MIN_PCIE_BW:-0}" != "0" ]; then
	      template_search_params="${template_search_params} pcie_bw>=${MIN_PCIE_BW}"
	    fi
	    echo "search_params: $template_search_params"
	    template_result=$(vastai create template \
	      --name "$ENDPOINT_NAME" \
	      --image "$FULL_IMAGE" \
	      --search_params "$template_search_params" \
	      --disk_space "$DISK_SIZE" \
	      --raw 2>/dev/null)

    template_hash=$(echo "$template_result" | jq -r '.template_hash // .hash // .hash_id // empty' 2>/dev/null | head -1 || true)
    if [ -z "${template_hash:-}" ] || [ "$template_hash" = "null" ]; then
      template_hash=$(vastai search templates "name=$ENDPOINT_NAME" --raw 2>/dev/null | jq -r 'sort_by(.created_at) | reverse | .[0].hash_id' 2>/dev/null | head -1 || true)
    fi
    if [ -z "${template_hash:-}" ] || [ "$template_hash" = "null" ]; then
      echo -e "${RED}Erreur: Impossible de recuperer le template hash${NC}"
      exit 1
    fi
    echo "Template hash: $template_hash"

    echo -e "\n${YELLOW}[2/3] Creation endpoint...${NC}"
    vastai create endpoint \
      --endpoint_name "$ENDPOINT_NAME" \
      --target_util 0.8 \
      --cold_mult 2.0 \
      --cold_workers "$MIN_WORKERS" \
      --max_workers "$MAX_WORKERS"

    sleep 2

    echo -e "\n${YELLOW}[3/3] Creation workergroup...${NC}"
    wg_launch_args="$(get_workergroup_launch_args)"
    echo "launch_args: $wg_launch_args"
    vastai create workergroup \
      --template_hash "$template_hash" \
      --endpoint_name "$ENDPOINT_NAME" \
      --cold_workers "$MIN_WORKERS" \
      --launch_args "$wg_launch_args"

    echo -e "\n${GREEN}Endpoint deploye!${NC}"
    show_usage_examples
    ;;

  down)
    echo -e "${YELLOW}=== Destroy Endpoint ===${NC}"
    check_requirements

    wg_ids=$(vastai show workergroups --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\") | .id" 2>/dev/null || true)
    if [ -n "${wg_ids:-}" ]; then
      for wg_id in $wg_ids; do
        vastai delete workergroup "$wg_id" 2>/dev/null || true
      done
      sleep 2
    fi

    endpoint_id=$(vastai show endpoints --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\") | .id" 2>/dev/null | head -1 || true)
    if [ -n "${endpoint_id:-}" ] && [ "$endpoint_id" != "null" ]; then
      vastai delete endpoint "$endpoint_id" 2>/dev/null || true
      echo -e "${GREEN}Endpoint supprime${NC}"
    else
      echo "Aucun endpoint trouve: $ENDPOINT_NAME"
    fi
    ;;

  status)
    check_requirements
    echo -e "${YELLOW}=== Status ===${NC}"
    echo "Endpoints:"
    endpoint_data=$(vastai show endpoints --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\")" 2>/dev/null || true)
    endpoint_line=$(echo "$endpoint_data" | jq -r "\"  id=\\(.id) state=\\(.endpoint_state) cold_workers=\\(.cold_workers) max_workers=\\(.max_workers) target_util=\\(.target_util)\"" 2>/dev/null | head -1 || true)
    if [ -n "${endpoint_line:-}" ] && [ "$endpoint_line" != "null" ]; then
      echo "$endpoint_line"
    else
      echo "  Aucun endpoint '$ENDPOINT_NAME'"
    fi

    echo ""
    echo "Workergroups:"
    wg_line=$(vastai show workergroups --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\") | \"  id=\\(.id) template_hash=\\(.template_hash) launch_args=\\(.launch_args)\" " 2>/dev/null | head -1 || true)
    if [ -n "${wg_line:-}" ]; then
      echo "$wg_line"
    else
      echo "  Aucun workergroup"
    fi

    # Health check detaille
    echo ""
    echo -e "${YELLOW}Model Status:${NC}"
    api_key=$(echo "$endpoint_data" | jq -r '.api_key' 2>/dev/null || true)
    if [ -n "${api_key:-}" ] && [ "$api_key" != "null" ]; then
      health_output=$(VAST_ENDPOINT_API_KEY="$api_key" ENDPOINT_NAME="$ENDPOINT_NAME" \
        python -u "$PROJECT_DIR/client.py" --health 2>&1) || true
      if [ -n "${health_output:-}" ]; then
        echo "$health_output" | grep -v "^Getting worker\|^Worker:" || true
      else
        echo "  Waiting for worker..."
      fi
    else
      echo "  Endpoint not ready (no API key)"
    fi
    ;;

  logs)
    check_requirements
    echo -e "${YELLOW}=== Endpoint Logs ===${NC}"
    endpoint_id=$(vastai show endpoints --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\") | .id" 2>/dev/null | head -1 || true)
    if [ -n "${endpoint_id:-}" ] && [ "$endpoint_id" != "null" ]; then
      vastai get endpt-logs "$endpoint_id" 2>&1 || true
    else
      echo "Aucun endpoint trouve: $ENDPOINT_NAME"
    fi
    ;;

  smoke)
    echo -e "${YELLOW}=== Smoke Test (health + generate) ===${NC}"
    check_requirements

    endpoint_data=$(vastai show endpoints --raw 2>/dev/null | jq -r ".[] | select(.endpoint_name==\"$ENDPOINT_NAME\")" 2>/dev/null || true)
    api_key=$(echo "$endpoint_data" | jq -r '.api_key' 2>/dev/null || true)
    if [ -z "${api_key:-}" ] || [ "$api_key" = "null" ]; then
      echo -e "${RED}Erreur: endpoint API key introuvable (endpoint pas deploye?)${NC}"
      exit 1
    fi

    echo -e "${YELLOW}[1/2] /health${NC}"
    VAST_ENDPOINT_API_KEY="$api_key" ENDPOINT_NAME="$ENDPOINT_NAME" \
      python -u "$PROJECT_DIR/client.py" --health || true

    echo -e "${YELLOW}[2/2] /generate${NC}"
    VAST_ENDPOINT_API_KEY="$api_key" ENDPOINT_NAME="$ENDPOINT_NAME" \
      python -u "$PROJECT_DIR/client.py" --prompt "A cute robot cat" --steps 8 --width 1024 --height 1024
    ;;

  clean-docker|docker-clean|prune)
    echo -e "${YELLOW}=== Docker Cleanup (prune) ===${NC}"
    echo -e "${YELLOW}This deletes unused images/containers/volumes to free disk space.${NC}"
    docker system df || true
    docker system prune -af --volumes
    docker system df || true
    ;;

  help|-h|--help|*)
    show_help
    ;;
esac
