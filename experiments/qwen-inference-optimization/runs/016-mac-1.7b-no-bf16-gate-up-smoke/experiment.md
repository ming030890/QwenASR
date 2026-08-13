# 016-mac-1.7b-no-bf16-gate-up-smoke: failed prototype

- Status: complete; implementation failure fixed in run 017
- Date (UTC): 2026-08-12
- Baseline: `015-mac-1.7b-prefill-scratch-ab`

## Hypothesis

Removing the owned BF16 gate/up fusion buffer further lowers 1.7B memory while
separate BF16 prefill projections preserve exact output.

## Result

The benchmark aborted before inference because the attempted reused up-projection
activation buffer was sized by decoder hidden width (2048) instead of FFN width
(6144). The bounds check prevented silent memory corruption.

## Decision

Reject this implementation, add a correctly sized reusable up-projection buffer,
and rerun under a new immutable experiment ID.

## Conclusion

Run 017 contains the corrected smoke and run 018 contains the accepted full
measurement. No performance or transcript claim is made from this failed run.
