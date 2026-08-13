# <ID>: <short experiment name>

- Status: planned
- Date (UTC): YYYY-MM-DD
- Platform: macOS / Android
- Backend: OpenBLAS (required)
- Git revision: `<commit>`
- Baseline: `<experiment id>`

## Hypothesis

What single change should improve which metric, and why?

## Change

Describe the code, build flag, model representation, or runtime setting changed.

## Fixed setup

- Model:
- Detected variant: 1.7B
- Model fingerprint:
- Input:
- Threads:
- Launches × runs:
- Prompt/language:
- Power and thermal state:

## Results

| Metric | Baseline | Candidate | Delta | Target met? |
| --- | ---: | ---: | ---: | --- |
| Median inference latency (ms) | | | | |
| Median model load time (ms) | | | | |
| Peak RSS (MiB) | | | | |
| Model size (MiB) | | | | |
| Runtime binary / Android package size (MiB) | | | | |

## Acceptance gates

- Exact transcript equality versus baseline: PASS / FAIL
- All baseline repetitions identical: PASS / FAIL
- All candidate repetitions identical: PASS / FAIL
- Same fixed audio inputs: PASS / FAIL

Any transcript mismatch—including capitalization, spacing, or punctuation—is an
automatic rejection regardless of performance improvement.

## Decision

Keep / reject / investigate.

## Conclusion

Explain the result, important caveats, and the next experiment in a few
sentences. A measurement without this conclusion is not a completed experiment.
