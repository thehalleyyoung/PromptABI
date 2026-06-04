from __future__ import annotations

import json
import time
from pathlib import Path

from promptabi import VerificationSession


def main() -> None:
    config = Path(__file__).parents[1] / "examples" / "minimal" / "promptabi.json"
    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        result = VerificationSession.from_config_file(config).run()
        if not result.ok:
            raise SystemExit("benchmark fixture unexpectedly failed")
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "benchmark": "skeleton-verify-smoke",
                "iterations": iterations,
                "seconds": round(elapsed, 6),
                "runs_per_second": round(iterations / elapsed, 2),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

