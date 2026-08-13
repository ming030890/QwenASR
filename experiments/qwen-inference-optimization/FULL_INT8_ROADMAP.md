# Full INT8 roadmap

## Current v15 mobile model inventory

Resume from the last accepted baseline,
`004-mac-openblas-v15-int8-keep-embeddings-baseline`, using model
`/Volumes/SSD/qwen-asr-m4-lmhead-int8-v15-lora-cp5000`. Do not use runs
005–011 as a baseline; they are rejected diagnostic candidates. Full baseline
identity and metrics are recorded in
[INT8_EMBEDDING_CONCLUSION.md](INT8_EMBEDDING_CONCLUSION.md#baseline-to-resume-from).

The accepted OpenBLAS baseline contains approximately:

- 568.4 MiB INT8 weights (decoder projections and tied LM head);
- 652.4 MiB BF16 weights;
- 297 MiB of the BF16 total in token embeddings;
- roughly 356 MiB of the BF16 total in the audio encoder and remaining tensors.

## Experiment outcome

Post-training symmetric and affine embedding quantizers were tested from group
size 4 through 16. Affine group 16 is implemented as a zero-copy mmap inference
format and saves 74.19 MiB (6.0%), but it changed three exact transcripts. A
logit trace showed both a genuine near tie (`占`/`佔`) and a more serious case
where quantization overturned a 2.05-logit baseline preference
(`仲要`/`重要`). BF16 embeddings therefore remain the accepted representation
until quantization-aware recovery is trained and validated.

See [INT8_EMBEDDING_CONCLUSION.md](INT8_EMBEDDING_CONCLUSION.md) for the model
format, measurements, logit evidence, and recovery recipe.

## Next experiments

1. Quantize audio-encoder projection and FFN weights independently while
   retaining BF16 embeddings and the accepted INT8 decoder/LM head.
2. Measure encoder latency, model size, load RSS, peak RSS, and exact transcripts
   against run `004` using OpenBLAS parity settings.
3. If exact equality is too strict for encoder quantization, do not silently
   relax it. Add a separately approved full-corpus CER/WER gate and report both
   exact-match and quality results.
4. Recover the tied embedding and LM head with fake-quantization-aware training
   and teacher-logit distillation. Do not use benchmark-specific token
   exceptions.
5. Export and validate the recovered affine group-16 model before extending
   fake quantization to encoder tensor families.
6. Full INT8 graduation requires an Android device gate after Mac screening.
