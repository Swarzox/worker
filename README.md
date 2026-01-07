# Z-Image Turbo - VAST.AI Serverless

Deploiement de Z-Image Turbo (Alibaba, 6B params) sur VAST.AI Serverless avec Cache-DiT.

## Configuration fixe (prod)

```
--cache --rdt 0.6 --scm fast --steps 8 --quantize-type float8 --compile
--warmup 2 --attn flash --height 1920 --width 1080
```

- Cache-DiT + FP8 quantization + torch.compile + FlashAttention-2
- Resolution fixe: 1080x1920 (portrait)
- 8 steps d'inference

## Structure

```
z-image-turbo-vastai/
├── Dockerfile           # Image app (code seulement)
├── Dockerfile.base      # Image base (deps + modele)
├── server.py           # Serveur FastAPI (inference)
├── worker.py           # Worker Vast.ai (serverless)
├── entrypoint.sh       # Entrypoint
├── docker-requirements.txt
├── client.py           # Client de test
└── scripts/deploy.sh   # Script de deploiement
```

## Prerequis

1. Compte Docker Hub
2. Compte Vast.ai avec credits
3. Token HuggingFace (1ere fois seulement)

```bash
cp .env.example .env
# Editer .env avec vos credentials
```

## Deploiement

```bash
# Build + Deploy
./scripts/deploy.sh deploy

# Ou separement:
./scripts/deploy.sh build-base  # Image base (rare)
./scripts/deploy.sh build       # Image app (rapide)
./scripts/deploy.sh up          # Deploy endpoint
```

## Commandes

```bash
./scripts/deploy.sh status   # Voir le status
./scripts/deploy.sh logs     # Voir les logs
./scripts/deploy.sh smoke    # Test (/health + /generate)
./scripts/deploy.sh down     # Supprimer l'endpoint
```

## Test client

```bash
export VAST_ENDPOINT_API_KEY="<endpoint_api_key>"
python client.py --health
python client.py --prompt "A cute robot cat"
```

## API

### GET/POST /health

```json
{
  "status": "healthy",
  "model_ready": true,
  "cache_dit_enabled": true,
  "cache_dit_rdt": 0.6,
  "compile_enabled": true
}
```

### POST /generate

Request (resolution fixe 1080x1920, 8 steps):
```json
{
  "prompt": "A cute robot cat",
  "width": 1080,
  "height": 1920,
  "num_inference_steps": 8,
  "seed": 0
}
```

Response:
```json
{
  "image": "base64...",
  "gpu_inference_ms": 1500.0
}
```

## Requirements

- GPU: RTX 4090 (compute capability >= 8.9 pour FP8)
- Disk: 150GB
- CUDA 12.6+
