"""Evaluation runner alias entrypoint.

Delegates execution to run_evaluation.py to avoid code duplication.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure evaluation directory is in sys.path
eval_dir = Path(__file__).resolve().parent
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))

from run_evaluation import (
    CONFIG_KEYS,
    DEFAULT_REQUEST_INTERVAL,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_COUNT,
    DEFAULT_RETRY_MAX_DELAY,
    run_evaluation,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-RAG retrieval and answer generation."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--mock",
        action="store_true",
        help="Run the deterministic offline fixture pipeline check.",
    )
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Run a live Gemini evaluation; requires GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of retrieved chunks (default: 5)."
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions.")
    parser.add_argument(
        "--configs", nargs="+", choices=CONFIG_KEYS,
        help="Configurations to run (default: all five).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_RETRY_COUNT,
        help="Retries after temporary 429 responses.",
    )
    parser.add_argument(
        "--retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY,
        help="Base exponential retry delay in seconds.",
    )
    parser.add_argument(
        "--retry-max-delay", type=float, default=DEFAULT_RETRY_MAX_DELAY,
        help="Maximum retry delay/wait budget in seconds.",
    )
    parser.add_argument(
        "--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL,
        help="Minimum seconds between runner-managed requests.",
    )
    args = parser.parse_args()

    run_evaluation(
        mock=args.mock,
        top_k=args.top_k,
        limit=args.limit,
        config_keys=args.configs,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
        request_interval=args.request_interval,
    )
