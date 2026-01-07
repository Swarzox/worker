#!/usr/bin/env python3
"""
Z-Image Turbo Client - Simplified
Usage: python client.py --prompt "A cute cat"
"""

import os
import argparse
import base64
import json
import time
import urllib.error
import urllib.request
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


# =============================================================================
# Config
# =============================================================================

_CACHE_FILE = Path("/tmp/zimage_workers.json")
_CACHE_TTL = 30  # seconds
_HEALTH_TIMEOUT = 2  # seconds

# Load tracking (in-memory, per-process)
_worker_loads = {}  # {url: count}

# Worker stats (persisted to file for cross-process visibility)
_STATS_FILE = Path("/tmp/zimage_stats.json")


def _load_dotenv():
    """Load .env file if present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ[key] = value


def _get_api_key(api_key: Optional[str] = None) -> str:
    """Get Vast.ai API key from arg, env, or CLI config."""
    if api_key:
        return api_key
    if os.getenv("VAST_API_KEY"):
        return os.getenv("VAST_API_KEY")
    try:
        raw = subprocess.check_output(["vastai", "show", "endpoints", "--raw"], text=True)
        endpoints = json.loads(raw)
        if endpoints:
            return endpoints[0].get("api_key", "")
    except Exception:
        pass
    raise RuntimeError("VAST_API_KEY not found. Set it in .env or run `vastai set api-key`.")


# =============================================================================
# Worker Discovery
# =============================================================================

def _check_worker_health(url: str) -> dict:
    """Check worker health status with full warmup details."""
    try:
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=_HEALTH_TIMEOUT) as resp:
            data = json.loads(resp.read())
            return {
                "url": url,
                "ready": data.get("status") == "healthy",
                "status": data.get("status", "unknown"),
                # Loading
                "step": data.get("loading_step", 0),
                "total_steps": data.get("loading_total_steps", 8),
                "loading_status": data.get("loading_status", ""),
                # Warmup
                "warmup_complete": data.get("warmup_complete", False),
                "warmup_in_progress": data.get("warmup_in_progress", False),
                "warmup_progress": data.get("warmup_progress", "0/0"),
                "warmup_done": data.get("warmup_done", 0),
                "warmup_total": data.get("warmup_total", 0),
                "warmup_elapsed": data.get("warmup_elapsed_seconds"),
                "warmup_first_time": data.get("warmup_first_inference_seconds"),
                "warmup_last_time": data.get("warmup_last_inference_seconds"),
                # Config
                "compile_enabled": data.get("compile_enabled", False),
                "cache_dit_enabled": data.get("cache_dit_enabled", False),
            }
    except Exception as e:
        return {"url": url, "ready": False, "status": "unreachable", "error": str(e)}


def _fetch_workers(api_key: str) -> list:
    """Fetch running workers from Vast.ai API."""
    url = f"https://console.vast.ai/api/v0/instances/?api_key={api_key}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    workers = []
    for inst in data.get("instances", []):
        ip = inst.get("public_ipaddr")
        port = inst.get("ports", {}).get("5001/tcp", [{}])[0].get("HostPort")
        if ip and port and inst.get("actual_status") == "running":
            workers.append({
                "url": f"http://{ip}:{port}",
                "id": inst.get("id")
            })
    return workers


def get_ready_workers(api_key: str, verbose: bool = True) -> list:
    """Get workers that are ready (model loaded). Cached for 30s."""
    # Check cache
    if _CACHE_FILE.exists():
        try:
            cache = json.loads(_CACHE_FILE.read_text())
            if time.time() - cache.get("ts", 0) < _CACHE_TTL:
                return cache.get("workers", [])
        except Exception:
            pass

    # Fetch from API
    try:
        workers = _fetch_workers(api_key)
    except Exception as e:
        print(f"[Warning] Could not fetch workers: {e}")
        return []

    if not workers:
        return []

    # Health check in parallel
    ready = []
    loading = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_check_worker_health, w["url"]): w for w in workers}
        for future in as_completed(futures, timeout=_HEALTH_TIMEOUT + 1):
            worker = futures[future]
            try:
                health = future.result()
                if health["ready"]:
                    ready.append(worker)
                elif health["status"] == "loading" and verbose:
                    loading.append(health)
            except Exception:
                pass

    # Show loading workers
    if verbose and loading:
        for h in loading:
            print(f"[Vast.ai] Worker loading [{h['step']}/{h['total_steps']}] {h['loading_status']}")

    # Cache result
    try:
        _CACHE_FILE.write_text(json.dumps({"ts": time.time(), "workers": ready}))
    except Exception:
        pass

    return ready


def _get_worker_load(url: str) -> int:
    """Get current load for a worker."""
    return _worker_loads.get(url, 0)


def _reserve_worker(url: str):
    """Reserve a slot on a worker."""
    _worker_loads[url] = _worker_loads.get(url, 0) + 1


def _release_worker(url: str):
    """Release a slot on a worker."""
    if url in _worker_loads:
        _worker_loads[url] = max(0, _worker_loads[url] - 1)


# =============================================================================
# Worker Stats (cross-process, file-based)
# =============================================================================

def _read_stats() -> dict:
    """Read worker stats from file."""
    try:
        if _STATS_FILE.exists():
            return json.loads(_STATS_FILE.read_text())
    except Exception:
        pass
    return {"workers": {}, "global": {"total_requests": 0, "total_success": 0, "total_fails": 0}}


def _write_stats(stats: dict):
    """Write worker stats to file."""
    try:
        _STATS_FILE.write_text(json.dumps(stats, indent=2))
    except Exception:
        pass


def _record_request(url: str, success: bool, duration: float = 0):
    """Record a request result for a worker."""
    stats = _read_stats()
    now = time.time()

    if url not in stats["workers"]:
        stats["workers"][url] = {
            "total_requests": 0,
            "success": 0,
            "fails": 0,
            "last_success": None,
            "last_fail": None,
            "last_request": None,
            "avg_duration": 0
        }

    w = stats["workers"][url]
    w["total_requests"] += 1
    w["last_request"] = now

    if success:
        w["success"] += 1
        w["last_success"] = now
        # Update rolling average duration
        if w["avg_duration"] == 0:
            w["avg_duration"] = duration
        else:
            w["avg_duration"] = (w["avg_duration"] * 0.8) + (duration * 0.2)
    else:
        w["fails"] += 1
        w["last_fail"] = now

    # Update global stats
    stats["global"]["total_requests"] += 1
    if success:
        stats["global"]["total_success"] += 1
    else:
        stats["global"]["total_fails"] += 1

    _write_stats(stats)


def _format_time_ago(ts: float) -> str:
    """Format timestamp as 'X ago'."""
    if ts is None:
        return "never"
    diff = time.time() - ts
    if diff < 60:
        return f"{int(diff)}s ago"
    elif diff < 3600:
        return f"{int(diff/60)}m ago"
    elif diff < 86400:
        return f"{int(diff/3600)}h ago"
    else:
        return f"{int(diff/86400)}d ago"


def _fetch_all_instances(api_key: str) -> list:
    """Fetch ALL instances (including stopped) from Vast.ai API."""
    url = f"https://console.vast.ai/api/v0/instances/?api_key={api_key}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data.get("instances", [])


def _format_datetime(ts: float) -> str:
    """Format timestamp as datetime string."""
    if ts is None:
        return "N/A"
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _parse_status_msg(msg: str) -> dict:
    """Parse status_msg to extract docker pull progress etc."""
    if not msg:
        return {"phase": "unknown", "detail": ""}

    msg_lower = msg.lower()

    if "pulling" in msg_lower:
        # Count layers being pulled
        layers = msg.count("Pulling fs layer")
        downloading = msg.count("Downloading")
        extracting = msg.count("Extracting")
        return {
            "phase": "docker_pull",
            "detail": f"{layers} layers, {downloading} downloading, {extracting} extracting"
        }
    elif "success" in msg_lower:
        return {"phase": "running", "detail": msg}
    elif "error" in msg_lower or "failed" in msg_lower:
        return {"phase": "error", "detail": msg}
    else:
        return {"phase": "other", "detail": msg[:100]}


def status(api_key: str = None):
    """Show detailed status of all workers."""
    if not api_key:
        api_key = _get_api_key()

    print("=" * 70)
    print("Z-IMAGE TURBO - DETAILED WORKER STATUS")
    print("=" * 70)

    # Fetch ALL instances from API
    try:
        instances = _fetch_all_instances(api_key)
    except Exception as e:
        print(f"Error fetching instances: {e}")
        return

    if not instances:
        print("No instances found.")
        return

    stats = _read_stats()

    # Summary counters
    counts = {"running": 0, "loading": 0, "exited": 0, "other": 0}
    total_cost_hr = 0

    print(f"\nFound {len(instances)} instance(s)\n")

    for inst in instances:
        inst_id = inst.get("id")
        actual = inst.get("actual_status") or "unknown"
        intended = inst.get("intended_status", "unknown")

        # Count by status
        if actual == "running":
            counts["running"] += 1
        elif actual == "loading":
            counts["loading"] += 1
        elif actual == "exited":
            counts["exited"] += 1
        else:
            counts["other"] += 1

        # Build URL
        ip = inst.get("public_ipaddr")
        port = inst.get("ports", {}).get("5001/tcp", [{}])[0].get("HostPort") if inst.get("ports") else None
        url = f"http://{ip}:{port}" if ip and port else None

        # Status icon
        if actual == "running":
            icon = "🟢"
        elif actual == "loading":
            icon = "🟡"
        elif actual == "exited":
            icon = "🔴"
        else:
            icon = "⚪"

        # Cost
        cost_hr = inst.get("dph_total", 0)
        gpu_cost = inst.get("instance", {}).get("gpuCostPerHour", 0)
        disk_cost = inst.get("instance", {}).get("diskHour", 0)
        total_cost_hr += cost_hr

        print(f"{icon} Instance {inst_id}")
        print(f"  ├─ Status: {actual.upper()} (intended: {intended})")
        print(f"  ├─ Location: {inst.get('geolocation', 'N/A')} [{inst.get('country_code', '??')}]")
        print(f"  ├─ GPU: {inst.get('gpu_name', 'N/A')} ({inst.get('gpu_ram', 0)/1024:.1f}GB VRAM)")
        print(f"  ├─ Cost: ${cost_hr:.3f}/hr (GPU: ${gpu_cost:.3f}, Disk: ${disk_cost:.3f})")
        print(f"  ├─ Created: {_format_datetime(inst.get('start_date'))}")

        # Status message (docker pull progress, etc)
        status_msg = inst.get("status_msg", "")
        parsed = _parse_status_msg(status_msg)
        if parsed["phase"] == "docker_pull":
            print(f"  ├─ Docker: PULLING ({parsed['detail']})")
        elif parsed["phase"] == "error":
            print(f"  ├─ Error: {parsed['detail'][:60]}")
        elif actual == "loading" and status_msg:
            print(f"  ├─ Loading: {status_msg[:80]}")

        # If running, check health endpoint
        if actual == "running" and url:
            print(f"  ├─ URL: {url}")
            health = _check_worker_health(url)

            if health["ready"]:
                # Model loaded - check warmup status
                if health.get("warmup_complete"):
                    print(f"  ├─ Health: ✓ READY + HOT (warmup complete)")
                    if health.get("warmup_last_time"):
                        print(f"  ├─ Perf: {health['warmup_last_time']:.2f}s/inference")
                elif health.get("warmup_in_progress"):
                    prog = health.get("warmup_progress", "?/?")
                    elapsed = health.get("warmup_elapsed")
                    elapsed_str = f" ({elapsed:.0f}s)" if elapsed else ""
                    print(f"  ├─ Health: ✓ READY + WARMING [{prog}]{elapsed_str}")
                    if health.get("warmup_first_time"):
                        print(f"  ├─ First inference: {health['warmup_first_time']:.2f}s (compiling)")
                    if health.get("warmup_last_time"):
                        print(f"  ├─ Latest inference: {health['warmup_last_time']:.2f}s")
                else:
                    print(f"  ├─ Health: ✓ READY (warmup not started)")

                # Config details
                if health.get("compile_enabled") or health.get("cache_dit_enabled"):
                    compile_str = "ON" if health.get("compile_enabled") else "OFF"
                    cache_str = "ON" if health.get("cache_dit_enabled") else "OFF"
                    print(f"  ├─ Config: compile={compile_str} cache_dit={cache_str}")
            elif health["status"] == "loading":
                step = health.get("step", 0)
                total = health.get("total_steps", 8)
                loading_status = health.get("loading_status", "")
                print(f"  ├─ Health: ⏳ LOADING [{step}/{total}] {loading_status}")
            else:
                print(f"  ├─ Health: ✗ UNREACHABLE ({health.get('error', 'unknown')[:40]})")

            # Request stats
            worker_stats = stats.get("workers", {}).get(url, {})
            if worker_stats:
                success = worker_stats.get("success", 0)
                fails = worker_stats.get("fails", 0)
                total_req = worker_stats.get("total_requests", 0)
                rate = (success / total_req * 100) if total_req > 0 else 0
                print(f"  ├─ Requests: {total_req} ({success}✓ {fails}✗) {rate:.0f}% success")
                print(f"  ├─ Last success: {_format_time_ago(worker_stats.get('last_success'))}")
                if worker_stats.get("last_fail"):
                    print(f"  ├─ Last fail: {_format_time_ago(worker_stats.get('last_fail'))}")
                if worker_stats.get("avg_duration"):
                    print(f"  ├─ Avg latency: {worker_stats['avg_duration']:.1f}s")
        elif actual == "exited":
            print(f"  ├─ Note: Instance stopped (GPU not billed, only disk ${disk_cost:.3f}/hr)")

        # GPU utilization (if available)
        gpu_util = inst.get("gpu_util")
        gpu_temp = inst.get("gpu_temp")
        if gpu_util is not None and actual == "running":
            print(f"  ├─ GPU Util: {gpu_util:.1f}%", end="")
            if gpu_temp:
                print(f" | Temp: {gpu_temp:.0f}°C", end="")
            print()

        print(f"  └─ Machine: {inst.get('machine_id')} | Host: {inst.get('host_id')}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Running:  {counts['running']}")
    print(f"Loading:  {counts['loading']}")
    print(f"Exited:   {counts['exited']}")
    print(f"Total:    {len(instances)}")
    print(f"\nTotal cost: ${total_cost_hr:.3f}/hr (${total_cost_hr*24:.2f}/day)")

    # Ready workers from cache
    ready_count = 0
    if _CACHE_FILE.exists():
        try:
            cache = json.loads(_CACHE_FILE.read_text())
            cache_age = time.time() - cache.get("ts", 0)
            ready_count = len(cache.get("workers", []))
            print(f"\nReady workers (cached): {ready_count} ({cache_age:.0f}s old)")
        except:
            pass

    # Global request stats
    g = stats.get("global", {})
    total_req = g.get("total_requests", 0)
    if total_req > 0:
        success = g.get("total_success", 0)
        fails = g.get("total_fails", 0)
        print(f"\nGlobal stats: {total_req} requests ({success}✓ {fails}✗) {success/total_req*100:.0f}% success")


def _next_worker(workers: list) -> dict:
    """Get worker with least load."""
    if not workers:
        return None
    # Sort by load, pick least loaded
    return min(workers, key=lambda w: _get_worker_load(w["url"]))


# =============================================================================
# Generation
# =============================================================================

def generate_direct(url: str, prompt: str, width=1024, height=1024, steps=8, seed=None) -> dict:
    """Generate image on a specific worker."""
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "seed": seed or 0
    }
    req = urllib.request.Request(
        f"{url}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    data["elapsed"] = time.time() - start
    return data


def generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = None,
    output_dir: str = "output",
    api_key: str = None,
    fal_key: str = None,
    use_fallback: bool = True,
    max_retries: int = 3
) -> str:
    """Generate image using Vast.ai workers with fal.ai fallback."""
    print(f"Generating: {prompt[:50]}...")
    print(f"Size: {width}x{height}, Steps: {steps}")

    workers = get_ready_workers(api_key)

    if not workers:
        print("[Vast.ai] No ready workers")
        if use_fallback:
            return generate_with_fal(prompt, width, height, steps, seed, output_dir, fal_key)
        raise Exception("No ready workers available")

    print(f"[Vast.ai] {len(workers)} ready worker(s)")

    last_error = None
    tried = set()

    for _ in range(min(max_retries, len(workers))):
        worker = _next_worker(workers)
        if worker["url"] in tried:
            continue
        tried.add(worker["url"])

        url = worker["url"]
        _reserve_worker(url)
        load = _get_worker_load(url)
        print(f"[Vast.ai] → {url} (load: {load})")

        try:
            data = generate_direct(url, prompt, width, height, steps, seed)

            # Save image
            Path(output_dir).mkdir(exist_ok=True)
            filename = f"{output_dir}/zimage_{data.get('seed', int(time.time()*1000))}.png"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(data["image"]))

            print(f"Done in {data['elapsed']:.1f}s")
            if data.get("gpu_inference_ms"):
                print(f"Inference: {data['gpu_inference_ms']:.1f}ms")
            print(f"Saved: {filename}")

            _release_worker(url)
            _record_request(url, success=True, duration=data['elapsed'])
            return filename

        except Exception as e:
            _release_worker(url)
            _record_request(url, success=False)
            last_error = str(e)
            print(f"[Vast.ai] Error: {last_error}")

    print(f"[Vast.ai] All workers failed")
    if use_fallback:
        return generate_with_fal(prompt, width, height, steps, seed, output_dir, fal_key)
    raise Exception(f"All workers failed: {last_error}")


def generate_with_fal(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = None,
    output_dir: str = "output",
    fal_key: str = None
) -> str:
    """Generate image via fal.ai."""
    fal_key = fal_key or os.getenv("FAL_KEY", "")
    if not fal_key:
        raise Exception("FAL_KEY not set")

    print(f"[fal.ai] Generating: {prompt[:50]}...")

    payload = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_inference_steps": min(steps, 8),
        "num_images": 1,
        "enable_safety_checker": False,
        "output_format": "png",
        "sync_mode": False
    }
    if seed:
        payload["seed"] = seed

    req = urllib.request.Request(
        "https://fal.run/fal-ai/z-image/turbo",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Key {fal_key}", "Content-Type": "application/json"},
        method="POST"
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - start

    # Download image
    image_url = result.get("images", [{}])[0].get("url")
    if not image_url:
        raise Exception("No image URL in response")

    with urllib.request.urlopen(image_url, timeout=60) as resp:
        image_data = resp.read()

    Path(output_dir).mkdir(exist_ok=True)
    filename = f"{output_dir}/zimage_fal_{result.get('seed', int(time.time()*1000))}.png"
    with open(filename, "wb") as f:
        f.write(image_data)

    print(f"[fal.ai] Done in {elapsed:.1f}s")
    if result.get("timings", {}).get("inference"):
        print(f"[fal.ai] Inference: {result['timings']['inference']*1000:.1f}ms")
    print(f"[fal.ai] Saved: {filename}")
    return filename


def generate_local(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = None,
    output_dir: str = "output",
    host: str = "127.0.0.1",
    port: int = 5001
) -> str:
    """Generate image using local server."""
    print(f"Generating: {prompt[:50]}...")
    url = f"http://{host}:{port}"

    data = generate_direct(url, prompt, width, height, steps, seed)

    Path(output_dir).mkdir(exist_ok=True)
    filename = f"{output_dir}/zimage_{data.get('seed', int(time.time()*1000))}.png"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(data["image"]))

    print(f"Done in {data['elapsed']:.1f}s")
    if data.get("gpu_inference_ms"):
        print(f"Inference: {data['gpu_inference_ms']:.1f}ms")
    print(f"Saved: {filename}")
    return filename


def health(host: str = "127.0.0.1", port: int = 5001):
    """Check local server health."""
    req = urllib.request.Request(f"http://{host}:{port}/health")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    if data.get("status") == "healthy":
        print("Status: healthy")
    else:
        step = data.get("loading_step", 0)
        total = data.get("loading_total_steps", 8)
        print(f"Status: loading [{step}/{total}] {data.get('loading_status', '')}")
    return data


# =============================================================================
# CLI
# =============================================================================

def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Z-Image Turbo Client")
    parser.add_argument("--prompt", help="Image prompt")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", default="output")
    parser.add_argument("--local", action="store_true", help="Use local server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--health", action="store_true", help="Check health")
    parser.add_argument("--status", action="store_true", help="Show worker status")
    parser.add_argument("--fal-only", action="store_true", help="Use fal.ai only")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fal.ai fallback")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--fal-key", default=os.getenv("FAL_KEY", ""))

    args = parser.parse_args()

    if args.health:
        health(args.host, args.port)
        return

    if args.status:
        api_key = _get_api_key(args.api_key) if args.api_key else None
        status(api_key)
        return

    if not args.prompt:
        print("Usage: python client.py --prompt 'A cute cat'")
        return

    if args.fal_only:
        generate_with_fal(args.prompt, args.width, args.height, args.steps,
                         args.seed, args.output, args.fal_key)
    elif args.local:
        generate_local(args.prompt, args.width, args.height, args.steps,
                      args.seed, args.output, args.host, args.port)
    else:
        try:
            api_key = _get_api_key(args.api_key)
            generate(args.prompt, args.width, args.height, args.steps,
                    args.seed, args.output, api_key, args.fal_key,
                    use_fallback=not args.no_fallback)
        except RuntimeError as e:
            print(str(e))
            print("Use --local for local server or --fal-only for fal.ai")


if __name__ == "__main__":
    main()
