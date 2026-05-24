# H1: macOS host baseline reference

Purpose:
- Run the same fixed benchmark inputs on this macOS host as a reference point
  for Android results. Do not mix these rows into `results-android.tsv`;
  host JSON is stored separately under `bench/host-results`.

Command shape:

```bash
PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
cargo build --release -p qwen-asr-cli --bin qwen-asr-bench-android

target/release/qwen-asr-bench-android \
  --model-dir /Volumes/SSD/qwen3-asr-0.6b \
  --input /private/tmp/qwen-asr-host-common-voice \
  --runs 3 \
  --threads 1 \
  > bench/host-results/host-osx-baseline-t1-r3-20260524-1520/common_voice-t1.json

target/release/qwen-asr-bench-android \
  --model-dir /Volumes/SSD/qwen3-asr-0.6b \
  --input bench/android-assets/ptt_guardrail.wav \
  --runs 3 \
  --threads 1 \
  > bench/host-results/host-osx-baseline-t1-r3-20260524-1520/ptt_guardrail-t1.json
```

Result:
- Label/path: `bench/host-results/host-osx-baseline-t1-r3-20260524-1520`.
- Platform flags: `["vDSP/Accelerate", "NEON", "DotProd"]`, target arch
  `aarch64`.
- Threads/runs: `1` thread, `3` runs.
- Common Voice median inference: `196.125 ms`.
- Per-clip inference: `cv_yue_000.wav` `188.842 ms`, `cv_yue_001.wav`
  `246.790 ms`, `cv_yue_002.wav` `252.131 ms`, `cv_yue_003.wav`
  `196.125 ms`, `cv_yue_004.wav` `187.569 ms`.
- Guardrail: `319.319 ms`, exact transcript
  `Hello,你好吗？已经坐低咗啦。`
- Compared with accepted Android dotprod
  `android-accept-dotprod-t1-r3-20260524-1348`, host Common Voice median is
  about `136.45x` faster (`26761.235 ms` Android vs `196.125 ms` host), and
  host guardrail is about `159.37x` faster (`50890.267 ms` Android vs
  `319.319 ms` host).

Caveats:
- This is a host reference, not an Android acceptance comparator. macOS uses
  Accelerate/vDSP and a very different memory/CPU stack.
- The first Common Voice file includes a slower cold inference run, but the
  recorded file result is the median of three runs.

Updated prompt-correct host reference:
- Label/path: `bench/host-results/host-osx-auto-zhprompt-t1-r3-20260524-1910`.
- Prompt/language: `--language Auto --prompt '音频是粤语，可能夹杂英语词或短语'`.
- Platform flags: `["vDSP/Accelerate", "NEON", "DotProd"]`, target arch
  `aarch64`.
- Threads/runs: `1` thread, `3` runs.
- Common Voice median inference: `262.911 ms`; median encode `39.634 ms`;
  median decode `224.362 ms`.
- Per-clip inference: `cv_yue_000.wav` `247.033 ms`, `cv_yue_001.wav`
  `308.042 ms`, `cv_yue_002.wav` `304.156 ms`, `cv_yue_003.wav`
  `241.333 ms`, `cv_yue_004.wav` `262.911 ms`.
- Guardrail: `378.397 ms`, transcript `哈喽，你好吗？已经坐低咗啦。`.
- Common Voice average CER under the current strict script/punctuation metric:
  `0.19168`.
