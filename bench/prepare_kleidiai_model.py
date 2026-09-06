#!/usr/bin/env python3
"""Create an I8MM-packed-only Android Qwen model for the KleidiAI backend.

The output base model retains shared tensors but omits every row-major INT8
matrix and scale represented in the packed shard. A second mmap-able
safetensors file contains the complete I8MM-packed decoder layout plus all
already-quantized encoder attention and FFN matrices.
"""

import argparse
import ctypes
import json
import shutil
import struct
from pathlib import Path

import numpy as np

from prepare_int8_model import get_tensor_np, quantize_bf16_to_int8, read_safetensors


DECODER_PACKABLE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_up_fused.weight",
    "mlp.down_proj.weight",
)
ENCODER_PACKABLE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.out_proj.weight",
    "fc1.weight",
    "fc2.weight",
)
ENCODER_PROJECTION_WEIGHTS = (
    "thinker.audio_tower.conv_out.weight",
    "thinker.audio_tower.proj1.weight",
    "thinker.audio_tower.proj2.weight",
)
PACKED_SUFFIX = "_kleidiai_i8mm_packed"


class PackParams(ctypes.Structure):
    _fields_ = [("lhs_zero_point", ctypes.c_int32), ("scale_multiplier", ctypes.c_float)]


def load_kleidiai(path: Path):
    lib = ctypes.CDLL(str(path))
    for name in (
        "kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm",
        "kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm",
        "kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm",
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


def is_decoder_packable(name: str) -> bool:
    return "thinker.model.layers." in name and name.endswith(DECODER_PACKABLE_SUFFIXES)


def is_encoder_packable(header: dict, name: str) -> bool:
    return (
        (
            "thinker.audio_tower.layers." in name
            and name.endswith(ENCODER_PACKABLE_SUFFIXES)
        )
        or name in ENCODER_PROJECTION_WEIGHTS
    ) and header[name]["dtype"] in ("BF16", "I8")


def is_packable(
    header: dict,
    name: str,
    pack_decoder: bool = True,
    pack_encoder: bool = True,
) -> bool:
    return (pack_decoder and is_decoder_packable(name)) or (
        pack_encoder and is_encoder_packable(header, name)
    )


def is_removed_row_major_tensor(
    header: dict,
    name: str,
    pack_decoder: bool,
    pack_encoder: bool,
) -> bool:
    if is_packable(header, name, pack_decoder, pack_encoder):
        return True
    scale_suffix = "_int8_scale"
    return name.endswith(scale_suffix) and is_packable(
        header, name[: -len(scale_suffix)], pack_decoder, pack_encoder
    )


def get_tensor_f32(header: dict, data: np.ndarray, name: str) -> np.ndarray:
    tensor = get_tensor_np(header, data, name)
    dtype = header[name]["dtype"]
    if dtype == "BF16":
        return (tensor.astype(np.uint32) << 16).view(np.float32)
    if dtype in ("F16", "F32"):
        return tensor.astype(np.float32)
    raise ValueError(f"expected floating-point tensor for {name}, got {dtype}")


def write_filtered_base_model(
    path: Path,
    header: dict,
    data: np.ndarray,
    pack_decoder: bool,
    pack_encoder: bool,
) -> tuple[int, int]:
    kept_names = [
        name
        for name in header
        if name != "__metadata__"
        and not is_removed_row_major_tensor(
            header, name, pack_decoder, pack_encoder
        )
    ]
    filtered_header = {}
    if "__metadata__" in header:
        filtered_header["__metadata__"] = header["__metadata__"]

    offset = 0
    for name in kept_names:
        meta = header[name]
        start, end = meta["data_offsets"]
        size = end - start
        filtered_header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size

    header_bytes = json.dumps(filtered_header, separators=(",", ":")).encode("utf-8")
    chunk_size = 64 * 1024 * 1024
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for name in kept_names:
            start, end = header[name]["data_offsets"]
            for chunk_start in range(start, end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end)
                stream.write(data[chunk_start:chunk_end].tobytes())

    return len(kept_names), len(header) - (1 if "__metadata__" in header else 0) - len(kept_names)


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for source in source_dir.iterdir():
        if not source.is_file() or "safetensors" in source.name:
            continue
        destination = output_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kleidiai-lib", required=True)
    parser.add_argument(
        "--pack-scope",
        choices=("all", "decoder", "encoder"),
        default="all",
        help="matrices to pack and remove from the base model (default: all)",
    )
    args = parser.parse_args()

    source_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.kleidiai.safetensors"
    base_output_path = output_dir / "model.safetensors"
    for path in (base_output_path, output_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    header, data = read_safetensors(str(source_dir / "model.safetensors"))
    pack_decoder = args.pack_scope in ("all", "decoder")
    pack_encoder = args.pack_scope in ("all", "encoder")
    names = sorted(
        name
        for name in header
        if name != "__metadata__"
        and is_packable(header, name, pack_decoder, pack_encoder)
    )
    if not names:
        raise SystemExit("no I8MM packing candidates found")

    lib = load_kleidiai(Path(args.kleidiai_lib).resolve())
    nr = lib.kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm()
    kr = lib.kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm()
    sr = lib.kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm()

    entries = []
    offset = 0
    for name in names:
        shape = header[name]["shape"]
        n = shape[0]
        k = int(np.prod(shape[1:]))
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
            source = get_tensor_np(header, data, name)
            if header[name]["dtype"] == "BF16":
                weights, scales = quantize_bf16_to_int8(source.reshape(n, k))
            else:
                weights = np.ascontiguousarray(source, dtype=np.int8)
                scales = np.ascontiguousarray(
                    get_tensor_np(header, data, name + "_int8_scale"),
                    dtype=np.float32,
                )
            bias_name = name.removesuffix("weight") + "bias"
            bias = (
                np.ascontiguousarray(get_tensor_f32(header, data, bias_name))
                if is_encoder_packable(header, name) and bias_name in header
                else np.zeros(n, dtype=np.float32)
            )
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

    kept, removed = write_filtered_base_model(
        base_output_path, header, data, pack_decoder, pack_encoder
    )
    copy_support_files(source_dir, output_dir)
    print(
        f"wrote {base_output_path} ({base_output_path.stat().st_size / 1048576:.2f} MiB; "
        f"kept {kept} tensors, removed {removed})"
    )
    print(f"wrote {output_path} ({output_path.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()
