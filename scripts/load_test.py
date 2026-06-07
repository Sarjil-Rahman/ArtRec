from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from urllib import request, error


def _post_recommend(
    base_url: str, api_key: str | None, user_id: str, limit: int
) -> tuple[bool, float, str | None]:
    payload = json.dumps({"user_id": user_id, "limit": limit}).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/recommend",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        req.add_header("X-API-Key", api_key)
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=30) as resp:
            resp.read()
            return 200 <= resp.status < 300, time.perf_counter() - started, None
    except error.HTTPError as exc:
        return False, time.perf_counter() - started, f"HTTP {exc.code}"
    except Exception as exc:
        return False, time.perf_counter() - started, str(exc)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple local load test for ArtRec /recommend."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--user-id", default="user_0000")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    latencies = []
    errors = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                _post_recommend, args.url, args.api_key, args.user_id, args.limit
            )
            for _ in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            ok, latency, err = future.result()
            latencies.append(latency)
            if not ok:
                errors.append(err)

    summary = {
        "total_requests": args.requests,
        "success_count": args.requests - len(errors),
        "error_count": len(errors),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "average_latency_ms": (
            round(statistics.mean(latencies) * 1000, 2) if latencies else 0.0
        ),
        "p50_latency_ms": round(_percentile(latencies, 50) * 1000, 2),
        "p95_latency_ms": round(_percentile(latencies, 95) * 1000, 2),
        "p99_latency_ms": round(_percentile(latencies, 99) * 1000, 2),
        "sample_errors": errors[:5],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
