"""
Z-Image Turbo - Vast.ai Worker
Worker serverless pour Vast.ai qui wrap le serveur FastAPI
"""

import math
try:
    from vastai_sdk import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig
except ImportError:  # Fallback compat
    from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig


def calculate_workload(params: dict) -> float:
    """
    Calcule le cout de workload pour la tarification.
    Formule basee sur la resolution et le nombre de steps.
    """
    width = params.get('width', 1024)
    height = params.get('height', 1024)
    num_inference_steps = params.get('num_inference_steps', 8)

    # Calcul base sur les tiles 512x512
    tiles_w = math.ceil(width / 512)
    tiles_h = math.ceil(height / 512)
    base_cost = tiles_w * tiles_h * 175 + 85

    # Ajustement par le nombre de steps (normalise sur 20)
    cost = base_cost * (num_inference_steps / 20)

    return float(cost)


def benchmark_generator():
    """Generate benchmark requests for performance testing."""
    return {
        'prompt': 'A cute robot cat',
        'width': 1024,
        'height': 1024,
        'num_inference_steps': 8
    }


def main():
    """Point d'entree du worker Vast.ai"""
    worker = Worker(
        WorkerConfig(
            model_server_url='http://127.0.0.1',
            model_server_port=5001,  # Port interne du serveur FastAPI
            model_log_file='/var/log/zimage/ready.log',
            # Note: Ne pas utiliser model_healthcheck_url car le vastai-sdk a un bug
            # qui empêche les healthchecks de fonctionner correctement
            handlers=[
                HandlerConfig(
                    route='/generate',
                    workload_calculator=calculate_workload,
                    benchmark_config=BenchmarkConfig(
                        generator=benchmark_generator,
                        runs=2,
                        concurrency=1
                    )
                ),
                HandlerConfig(route='/health'),
                HandlerConfig(route='/queue-status'),
            ],
            log_action_config=LogActionConfig(
                on_load=['Model loaded and ready for inference']
            )
        )
    )

    worker.run()


if __name__ == '__main__':
    main()
