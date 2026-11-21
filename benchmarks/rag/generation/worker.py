"""Fresh local generator worker; semantic judging happens only in the parent."""

from benchmarks.common.process import json_worker_main
from benchmarks.rag.generation.evaluate import execute_payload

if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute_payload))
