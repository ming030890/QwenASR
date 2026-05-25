#!/usr/bin/env python3
"""Prepare INT8-decoder model for Q2 experiment.

Reads the BF16 model.safetensors, replaces all decoder layer projection weights
(q/k/v/o attention, gate/up/down MLP) with INT8 + per-row F32 scales, and
writes the result to OUTPUT_DIR/model.safetensors.

All non-decoder tensors (encoder weights, embeddings, norms, etc.) are copied
unchanged as BF16.  Vocabulary files (vocab.json, merges.txt) are symlinked or
copied as-is.

The resulting model file is ~380 MB smaller than the original, eliminating
on-device BF16→INT8 quantization at model-load time.

Naming convention in the output file:
  {layer}.{proj}.weight          dtype I8   [out_dim, in_dim]   (replaces BF16)
  {layer}.{proj}.weight_int8_scale  dtype F32  [out_dim]         (new)

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

# Suffixes of per-layer decoder projection weights to replace with INT8.
DECODER_PROJ_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

DTYPE_SIZES = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "I32": 4, "I64": 8, "BOOL": 1}


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

def write_safetensors(path: str, tensors: list[tuple[str, str, np.ndarray]]) -> None:
    """Write tensors to a safetensors file.

    tensors: list of (name, dtype_str, data_ndarray) in desired layout order.
    dtype_str must be one of the standard safetensors dtype strings (F32, BF16, I8, …).
    """
    header_entries = {}
    offset = 0
    flat_bufs = []

    for name, dtype_str, arr in tensors:
        raw = arr.tobytes()
        n_bytes = len(raw)
        header_entries[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="/Volumes/SSD/qwen3-asr-0.6b",
                    help="Source model directory (contains model.safetensors)")
    ap.add_argument("--output-dir", default="/Volumes/SSD/qwen-asr-int8dec",
                    help="Output directory for INT8 model")
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
    print(f"  decoder projection tensors to quantize: {len(decoder_proj_names)}", flush=True)

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
        else:
            # Copy as-is: get raw bytes and wrap in uint8 ndarray
            start, end = meta["data_offsets"]
            raw = np.array(data_mmap[start:end], dtype=np.uint8)
            out_tensors.append((name, dtype_str, raw))

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
