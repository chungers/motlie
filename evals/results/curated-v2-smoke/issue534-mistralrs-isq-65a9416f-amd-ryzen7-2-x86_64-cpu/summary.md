# Eval Matrix Run

- issue: `534`
- snapshot: `curated-v2-smoke`
- profile: `local-cpu-x86_64`
- launched cells: `1`
- pre-run records: `0`
- note: mistralrs CPU ISQ q4 direct rerun from committed head `65a9416f`; row validated with `backend_observation`. A q8 direct-run probe on this host ended with `scenario exceeded --max-wall-time-secs=1200s wall-time backstop` before writing a JSONL result row, so no q8 metric is committed.
