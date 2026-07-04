#!/usr/bin/env python3
"""Prepare INT8-decoder model for Q2 experiment.

Reads the BF16 model.safetensors, replaces all decoder layer projection weights
(q/k/v/o attention, gate/up/down MLP) with INT8 + per-row F32 scales, adds a
pre-quantized tied lm_head tensor, and writes the result to
OUTPUT_DIR/model.safetensors.

All non-decoder tensors (encoder weights, embeddings, norms, etc.) are copied
unchanged as BF16.  Vocabulary files (vocab.json, merges.txt) are symlinked or
copied as-is.

The resulting model file is ~380 MB smaller than the original, eliminating
on-device BF16→INT8 quantization at model-load time.

Naming convention in the output file:
  {layer}.{proj}.weight          dtype I8   [out_dim, in_dim]   (replaces BF16)
  {layer}.{proj}.weight_int8_scale  dtype F32  [out_dim]         (new)
  thinker.lm_head.weight         dtype I8   [vocab, hidden]     (new/replaced tied head)
  thinker.lm_head.weight_int8_scale dtype F32 [vocab]           (new)

Usage:
  bench/prepare_int8_model.py \\
      --model-dir /Volumes/SSD/qwen3-asr-0.6b \\
      --output-dir /Volumes/SSD/qwen-asr-int8dec
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

# Suffixes of per-layer decoder projection weights to quantize individually.
# gate_proj and up_proj are handled separately: they are fused into gate_up_fused.weight.
DECODER_PROJ_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
]

DTYPE_SIZES = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "I32": 4, "I64": 8, "BOOL": 1}
EMBED_TOKENS = "thinker.model.embed_tokens.weight"
LM_HEAD = "thinker.lm_head.weight"
LM_HEAD_SCALE = "thinker.lm_head.weight_int8_scale"


# ---------------------------------------------------------------------------
# Minimal safetensors reader (no dependencies beyond numpy)
# ---------------------------------------------------------------------------

def read_safetensors(path: str) -> tuple[dict, np.ndarray]:
    """Return (header_dict, data_mmap) for a safetensors file.

    header_dict maps tensor_name → {dtype, shape, data_offsets: [start, end]}.
    data_mmap is a numpy uint8 memmap covering the data section.
    """
    f = open(path, "rb")
    header_size = struct.unpack("<Q", f.read(8))[0]
    header_bytes = f.read(header_size)
    header = json.loads(header_bytes.decode("utf-8"))
    f.close()

    data_offset_in_file = 8 + header_size
    data_mmap = np.memmap(path, dtype=np.uint8, mode="r", offset=data_offset_in_file)
    return header, data_mmap


def get_tensor_np(header: dict, data_mmap: np.ndarray, name: str) -> np.ndarray:
    """Return a numpy array view for a named tensor."""
    meta = header[name]
    dtype_str = meta["dtype"]
    shape = meta["shape"]
    start, end = meta["data_offsets"]

    dtype_map = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.uint16,  # treat raw bits
        "I8": np.int8,
        "I32": np.int32,
        "I64": np.int64,
        "BOOL": np.bool_,
    }
    np_dtype = dtype_map[dtype_str]
    raw = data_mmap[start:end]
    return raw.view(np_dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_bf16_to_int8(w_bf16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize BF16 weight matrix to INT8 per-row with F32 scales.

    w_bf16: uint16 ndarray with shape [out_dim, in_dim] (raw BF16 bits).
    Returns (int8_weights, scales) with shapes [out_dim, in_dim] and [out_dim].

    Matches the Rust generic path in kernels/mod.rs:quantize_bf16_weights_to_int8.
    The NEON path uses vcvtq_s32_f32 (truncate) while the scalar path uses
    round(); we use round() here to match the scalar fallback and stay within
    ±1 LSB of the NEON result.
    """
    assert w_bf16.dtype == np.uint16
    out_dim, in_dim = w_bf16.shape

    # Convert BF16 bit patterns to F32
    w_f32 = w_bf16.astype(np.uint32) << 16
    w_f32 = w_f32.view(np.float32)

    max_abs = np.max(np.abs(w_f32), axis=1)  # [out_dim]
    scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
    inv_scales = (127.0 / np.maximum(max_abs, 1e-10)).astype(np.float32)  # [out_dim]

    # Quantize: round(v * inv_scale).clamp(-127, 127) as i8
    q = np.round(w_f32 * inv_scales[:, np.newaxis]).clip(-127, 127).astype(np.int8)
    return q, scales


# ---------------------------------------------------------------------------
# Safetensors writer
# ---------------------------------------------------------------------------

def write_safetensors(path: str, tensors: list[tuple]) -> None:
    """Write tensors to a safetensors file.

    tensors: list of (name, dtype_str, data_ndarray [, shape_override]) tuples.
    dtype_str must be one of the standard safetensors dtype strings (F32, BF16, I8, …).
    shape_override: if provided (as a list of ints), written to the header instead of arr.shape.
    Used for raw-byte copies of non-native-dtype arrays where arr.shape would be [n_bytes].
    """
    header_entries = {}
    offset = 0
    flat_bufs = []

    for entry in tensors:
        name, dtype_str, arr = entry[0], entry[1], entry[2]
        shape = list(entry[3]) if len(entry) > 3 else list(arr.shape)
        raw = arr.tobytes()
        n_bytes = len(raw)
        header_entries[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + n_bytes],
        }
        offset += n_bytes
        flat_bufs.append(raw)

    header_json = json.dumps(header_entries, separators=(",", ":")).encode("utf-8")
    header_size = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        for buf in flat_bufs:
            f.write(buf)

    print(f"  wrote {path}  ({os.path.getsize(path) // (1024*1024)} MB)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def is_decoder_proj(name: str) -> bool:
    if "thinker.model.layers." not in name:
        return False
    return any(name.endswith(sfx) for sfx in DECODER_PROJ_SUFFIXES)

def is_decoder_gate_proj(name: str) -> bool:
    return "thinker.model.layers." in name and name.endswith("mlp.gate_proj.weight")

def is_decoder_up_proj(name: str) -> bool:
    return "thinker.model.layers." in name and name.endswith("mlp.up_proj.weight")


def has_untied_lm_head(header: dict) -> bool:
    """Return true when the source model has a separate BF16 lm_head tensor."""
    if LM_HEAD not in header:
        return False
    lm_shape = header[LM_HEAD]["shape"]
    embed_shape = header.get(EMBED_TOKENS, {}).get("shape")
    return lm_shape != embed_shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="/Volumes/SSD/qwen3-asr-0.6b",
                    help="Source model directory (contains model.safetensors)")
    ap.add_argument("--output-dir", default="/Volumes/SSD/qwen-asr-int8dec",
                    help="Output directory for INT8 model")
    ap.add_argument("--drop-embed-tokens", action="store_true",
                    help="Drop BF16 thinker.model.embed_tokens.weight and use INT8 lm_head for token embeddings")
    args = ap.parse_args()

    src = Path(args.model_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    model_src = str(src / "model.safetensors")
    model_dst = str(dst / "model.safetensors")

    if not os.path.exists(model_src):
        sys.exit(f"model.safetensors not found in {src}")

    print(f"Reading {model_src} …", flush=True)
    header, data_mmap = read_safetensors(model_src)

    # Collect tensor names (preserve order from header, skip __metadata__)
    names = [k for k in header if k != "__metadata__"]

    # Identify decoder projection tensors
    decoder_proj_names = {n for n in names if is_decoder_proj(n)}
    gate_proj_names = {n for n in names if is_decoder_gate_proj(n)}
    quantize_tied_lm_head = not has_untied_lm_head(header)
    print(f"  individual projections to quantize: {len(decoder_proj_names)}, "
          f"gate+up pairs to fuse: {len(gate_proj_names)}", flush=True)
    print(f"  tied lm_head pre-quantization: {'yes' if quantize_tied_lm_head else 'no'}",
          flush=True)
    print(f"  drop BF16 token embeddings: {'yes' if args.drop_embed_tokens else 'no'}",
          flush=True)

    # Build the output tensor list
    out_tensors: list[tuple[str, str, np.ndarray]] = []
    quantized_count = 0

    for name in names:
        meta = header[name]
        dtype_str = meta["dtype"]

        if name in decoder_proj_names:
            assert dtype_str == "BF16", f"Expected BF16 for {name}, got {dtype_str}"
            w_bf16 = get_tensor_np(header, data_mmap, name)
            assert w_bf16.ndim == 2, f"Expected 2D weight for {name}"
            print(f"  quantizing {name} {w_bf16.shape} …", flush=True)
            w_int8, scales = quantize_bf16_to_int8(w_bf16)
            out_tensors.append((name, "I8", w_int8))
            out_tensors.append((name + "_int8_scale", "F32", scales))
            quantized_count += 1
        elif is_decoder_gate_proj(name):
            # Fuse gate + up into a single pre-interleaved tensor so the Rust
            # loader can point directly into the mmap region without copying.
            assert dtype_str == "BF16", f"Expected BF16 for {name}, got {dtype_str}"
            up_name = name.replace("gate_proj", "up_proj")
            gate_bf16 = get_tensor_np(header, data_mmap, name)
            up_bf16 = get_tensor_np(header, data_mmap, up_name)
            assert gate_bf16.shape == up_bf16.shape, \
                f"gate/up shape mismatch: {gate_bf16.shape} vs {up_bf16.shape}"
            inter, hidden = gate_bf16.shape
            print(f"  fusing+quantizing gate+up for {name[:-len('gate_proj.weight')]} "
                  f"[{inter},{hidden}] …", flush=True)
            gate_int8, gate_scales = quantize_bf16_to_int8(gate_bf16)
            up_int8, up_scales = quantize_bf16_to_int8(up_bf16)
            # Interleave rows: [gate0, up0, gate1, up1, ...]
            fused_int8 = np.empty((2 * inter, hidden), dtype=np.int8)
            fused_int8[0::2] = gate_int8
            fused_int8[1::2] = up_int8
            fused_scales = np.empty(2 * inter, dtype=np.float32)
            fused_scales[0::2] = gate_scales
            fused_scales[1::2] = up_scales
            fused_name = name.replace("gate_proj.weight", "gate_up_fused.weight")
            out_tensors.append((fused_name, "I8", fused_int8))
            out_tensors.append((fused_name + "_int8_scale", "F32", fused_scales))
            quantized_count += 1  # gate + up emitted as a single fused tensor
        elif is_decoder_up_proj(name):
            pass  # already emitted as gate_up_fused when gate_proj was processed
        elif name == LM_HEAD and quantize_tied_lm_head:
            assert dtype_str == "BF16", f"Expected BF16 for {name}, got {dtype_str}"
            lm_bf16 = get_tensor_np(header, data_mmap, name)
            assert lm_bf16.ndim == 2, f"Expected 2D weight for {name}"
            print(f"  quantizing tied lm_head from {name} {lm_bf16.shape} …", flush=True)
            lm_int8, lm_scales = quantize_bf16_to_int8(lm_bf16)
            out_tensors.append((LM_HEAD, "I8", lm_int8))
            out_tensors.append((LM_HEAD_SCALE, "F32", lm_scales))
            quantized_count += 1
        elif name == EMBED_TOKENS and args.drop_embed_tokens and quantize_tied_lm_head:
            pass
        else:
            # Copy as-is: raw bytes wrapped as uint8 ndarray.
            # Pass original shape explicitly — arr.shape would be [n_bytes], not the
            # tensor's logical shape, which would corrupt numel() in the Rust reader.
            start, end = meta["data_offsets"]
            raw = np.array(data_mmap[start:end], dtype=np.uint8)
            out_tensors.append((name, dtype_str, raw, list(meta["shape"])))

        if name == EMBED_TOKENS and quantize_tied_lm_head and LM_HEAD not in header:
            assert dtype_str == "BF16", f"Expected BF16 for {name}, got {dtype_str}"
            embed_bf16 = get_tensor_np(header, data_mmap, name)
            assert embed_bf16.ndim == 2, f"Expected 2D weight for {name}"
            print(f"  quantizing tied lm_head from {name} {embed_bf16.shape} …", flush=True)
            lm_int8, lm_scales = quantize_bf16_to_int8(embed_bf16)
            out_tensors.append((LM_HEAD, "I8", lm_int8))
            out_tensors.append((LM_HEAD_SCALE, "F32", lm_scales))
            quantized_count += 1

    print(f"\nQuantized {quantized_count} tensors.", flush=True)
    print(f"Writing {model_dst} …", flush=True)
    write_safetensors(model_dst, out_tensors)

    # Copy vocabulary files
    for fname in ("vocab.json", "merges.txt"):
        src_f = src / fname
        dst_f = dst / fname
        if src_f.exists():
            shutil.copy2(str(src_f), str(dst_f))
            print(f"  copied {fname}", flush=True)

    src_mb = os.path.getsize(model_src) // (1024 * 1024)
    dst_mb = os.path.getsize(model_dst) // (1024 * 1024)
    print(f"\nDone. Original: {src_mb} MB  →  INT8 model: {dst_mb} MB  "
          f"(saved {src_mb - dst_mb} MB)", flush=True)


if __name__ == "__main__":
    main()
