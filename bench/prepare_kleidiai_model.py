#!/usr/bin/env python3
"""Create a Qwen model directory with offline KleidiAI-packed decoder weights.

The source row-major INT8 tensors remain available for the existing one-token
decode kernel. A second mmap-able safetensors file contains the DotProd-packed
layout used by decoder prefill, avoiding all runtime RHS packing.
"""

import argparse
import ctypes
import json
import os
import struct
from pathlib import Path

import numpy as np

from prepare_int8_model import get_tensor_np, read_safetensors


PACKABLE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_up_fused.weight",
    "mlp.down_proj.weight",
)
PACKED_SUFFIX = "_kleidiai_dotprod_packed"


class PackParams(ctypes.Structure):
    _fields_ = [("lhs_zero_point", ctypes.c_int32), ("scale_multiplier", ctypes.c_float)]


def load_kleidiai(path: Path):
    lib = ctypes.CDLL(str(path))
    for name in (
        "kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod",
        "kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod",
        "kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod",
    ):
        getattr(lib, name).restype = ctypes.c_size_t
    lib.kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.restype = ctypes.c_size_t
    lib.kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(PackParams),
    ]
    return lib


def is_packable(name: str) -> bool:
    return "thinker.model.layers." in name and name.endswith(PACKABLE_SUFFIXES)


def link_source_files(source_dir: Path, output_dir: Path) -> None:
    for source in source_dir.iterdir():
        if not source.is_file() or source.name == "model.kleidiai.safetensors":
            continue
        destination = output_dir / source.name
        if not destination.exists():
            destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kleidiai-lib", required=True)
    args = parser.parse_args()

    source_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.kleidiai.safetensors"
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")

    header, data = read_safetensors(str(source_dir / "model.safetensors"))
    names = sorted(name for name in header if name != "__metadata__" and is_packable(name))
    if not names:
        raise SystemExit("no packed-decoder candidates found")

    lib = load_kleidiai(Path(args.kleidiai_lib).resolve())
    nr = lib.kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod()
    kr = lib.kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod()
    sr = lib.kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod()

    entries = []
    offset = 0
    for name in names:
        n, k = header[name]["shape"]
        size = lib.kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, nr, kr, sr)
        packed_name = name + PACKED_SUFFIX
        entries.append((name, packed_name, n, k, size, offset))
        offset += size

    packed_header = {
        packed_name: {"dtype": "I8", "shape": [size], "data_offsets": [start, start + size]}
        for _, packed_name, _, _, size, start in entries
    }
    header_bytes = json.dumps(packed_header, separators=(",", ":")).encode("utf-8")
    params = PackParams(1, 1.0)

    with output_path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for index, (name, _, n, k, size, _) in enumerate(entries, 1):
            weights = np.ascontiguousarray(get_tensor_np(header, data, name), dtype=np.int8)
            scales = np.ascontiguousarray(
                get_tensor_np(header, data, name + "_int8_scale"), dtype=np.float32
            )
            bias = np.zeros(n, dtype=np.float32)
            packed = np.empty(size, dtype=np.uint8)
            lib.kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
                1,
                n,
                k,
                nr,
                kr,
                sr,
                weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_void_p(packed.ctypes.data),
                0,
                ctypes.byref(params),
            )
            stream.write(packed.tobytes())
            print(f"[{index:3d}/{len(entries)}] {name}: {size / 1048576:.2f} MiB", flush=True)

    link_source_files(source_dir, output_dir)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()
