#!/usr/bin/env python3
"""Stream a sharded BF16 Qwen3-ASR model into one mmap-ready INT8 model.

Decoder attention/FFN projections and the tied LM head are quantized per output
row. Token embeddings and the audio encoder remain BF16 by default. With
``--drop-embed-tokens``, token lookup reuses the tied INT8 LM head instead of
retaining a duplicate BF16 embedding matrix. ``--int8-encoder-ffn`` additionally
quantizes audio-encoder FC1/FC2 weights per output row. Conversion is bounded to
a few tensor-row chunks; neither the source nor destination model is held in
RAM.
"""

import argparse
from collections import defaultdict
import json
import re
import shutil
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from prepare_int8_model import (
    DTYPE_SIZES,
    EMBED_TOKENS,
    LM_HEAD,
    LM_HEAD_SCALE,
    is_decoder_gate_proj,
    is_decoder_proj,
    is_decoder_up_proj,
)


@dataclass
class SourceTensor:
    path: Path
    data_start: int
    dtype: str
    shape: list[int]
    start: int
    end: int

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass
class LoRAProjection:
    a: np.ndarray
    b: np.ndarray
    scale: float


@dataclass
class OutputTensor:
    name: str
    dtype: str
    shape: list[int]
    kind: str
    sources: tuple[str, ...]

    @property
    def size(self) -> int:
        result = DTYPE_SIZES[self.dtype]
        for dim in self.shape:
            result *= dim
        return result


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return header, 8 + header_size


def load_sources(model_dir: Path) -> tuple[list[str], dict[str, SourceTensor]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        names = list(weight_map)
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ["model.safetensors"]
        header, _ = read_header(model_dir / shard_names[0])
        names = [name for name in header if name != "__metadata__"]
        weight_map = {name: shard_names[0] for name in names}

    shard_headers = {}
    for shard_name in shard_names:
        shard_headers[shard_name] = read_header(model_dir / shard_name)

    sources = {}
    for name in names:
        shard_name = weight_map[name]
        header, data_start = shard_headers[shard_name]
        meta = header[name]
        start, end = meta["data_offsets"]
        sources[name] = SourceTensor(
            path=model_dir / shard_name,
            data_start=data_start,
            dtype=meta["dtype"],
            shape=list(meta["shape"]),
            start=start,
            end=end,
        )
    return names, sources


LORA_KEY_RE = re.compile(r"^(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$")


def adapter_target_to_base(name: str) -> str:
    for prefix in ("base_model.model.",):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if name.startswith("thinker."):
        return name + ".weight"
    return "thinker." + name + ".weight"


def load_lora_adapter(adapter_dir: Path) -> dict[str, LoRAProjection]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if config.get("use_dora"):
        raise NotImplementedError("direct INT8 export does not support DoRA adapters")
    if config.get("fan_in_fan_out"):
        raise NotImplementedError("direct INT8 export does not support fan_in_fan_out")
    if (adapter_dir / "mega_lora_blocks.json").exists():
        raise NotImplementedError("direct INT8 export does not support block-composed LoRA")

    path = adapter_dir / "adapter_model.safetensors"
    header, data_start = read_header(path)
    grouped = defaultdict(dict)
    raw_names = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        match = LORA_KEY_RE.match(key)
        if not match:
            continue
        raw_name, which = match.groups()
        if meta["dtype"] != "F32":
            raise ValueError(f"adapter tensor must be F32, got {meta['dtype']}: {key}")
        start, end = meta["data_offsets"]
        tensor = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            offset=data_start + start,
            shape=tuple(meta["shape"]),
        )
        base_name = adapter_target_to_base(raw_name)
        grouped[base_name][which] = np.asarray(tensor)
        raw_names[base_name] = raw_name

    default_rank = config.get("r")
    default_alpha = config.get("lora_alpha", default_rank)
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    projections = {}
    for base_name, pair in grouped.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"incomplete adapter pair for {base_name}: {sorted(pair)}")
        raw_name = raw_names[base_name]
        short_name = raw_name.removeprefix("base_model.model.")
        rank = rank_pattern.get(raw_name, rank_pattern.get(short_name, default_rank))
        if rank is None:
            rank = pair["A"].shape[0]
        alpha = alpha_pattern.get(raw_name, alpha_pattern.get(short_name, default_alpha))
        projections[base_name] = LoRAProjection(
            a=pair["A"],
            b=pair["B"],
            scale=float(alpha) / float(rank),
        )
    if not projections:
        raise ValueError(f"no LoRA A/B pairs found in {path}")
    return projections


def is_encoder_ffn_weight(name: str) -> bool:
    return (
        name.startswith("thinker.audio_tower.layers.")
        and (name.endswith(".fc1.weight") or name.endswith(".fc2.weight"))
    )


def is_encoder_attention_weight(name: str) -> bool:
    return name.startswith("thinker.audio_tower.layers.") and any(
        name.endswith(suffix)
        for suffix in (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.out_proj.weight",
        )
    )


def encoder_layer_index(name: str) -> int:
    return int(name.split("thinker.audio_tower.layers.", 1)[1].split(".", 1)[0])


def build_plan(
    names: list[str],
    sources: dict[str, SourceTensor],
    drop_embed_tokens: bool = False,
    int8_encoder_ffn: bool = False,
    int8_encoder_attention: bool = False,
    encoder_ffn_group_size: int | None = None,
    encoder_ffn_layer_start: int = 0,
    encoder_ffn_layer_limit: int | None = None,
    encoder_attention_layer_start: int | None = None,
    encoder_attention_layer_limit: int | None = None,
) -> list[OutputTensor]:
    attention_layer_start = (
        encoder_ffn_layer_start
        if encoder_attention_layer_start is None
        else encoder_attention_layer_start
    )
    attention_layer_limit = (
        encoder_ffn_layer_limit
        if encoder_attention_layer_limit is None
        else encoder_attention_layer_limit
    )
    plan = []
    has_lm_head = LM_HEAD in sources
    if drop_embed_tokens and has_lm_head:
        embed = sources[EMBED_TOKENS]
        lm_head = sources[LM_HEAD]
        if embed.shape != lm_head.shape:
            raise ValueError(
                "cannot reuse an untied LM head for token lookup: "
                f"embedding shape {embed.shape} != LM-head shape {lm_head.shape}"
            )
    for name in names:
        source = sources[name]
        quantize_encoder_ffn = (
            int8_encoder_ffn
            and is_encoder_ffn_weight(name)
            and encoder_layer_index(name) >= encoder_ffn_layer_start
            and (
                encoder_ffn_layer_limit is None
                or encoder_layer_index(name) < encoder_ffn_layer_limit
            )
        )
        quantize_encoder_attention = (
            int8_encoder_attention
            and is_encoder_attention_weight(name)
            and encoder_layer_index(name) >= attention_layer_start
            and (
                attention_layer_limit is None
                or encoder_layer_index(name) < attention_layer_limit
            )
        )
        if quantize_encoder_attention:
            assert source.dtype == "BF16" and len(source.shape) == 2
            plan.append(OutputTensor(name, "I8", source.shape, "quant", (name,)))
            plan.append(
                OutputTensor(
                    name + "_int8_scale", "F32", [source.shape[0]], "scale", (name,)
                )
            )
        elif quantize_encoder_ffn:
            assert source.dtype == "BF16" and len(source.shape) == 2
            if encoder_ffn_group_size:
                if source.shape[1] % encoder_ffn_group_size != 0:
                    raise ValueError(
                        f"{name} input dim {source.shape[1]} is not divisible by "
                        f"group size {encoder_ffn_group_size}"
                    )
                scale_shape = [
                    source.shape[0],
                    source.shape[1] // encoder_ffn_group_size,
                ]
                quant_kind = f"group_quant:{encoder_ffn_group_size}"
                scale_kind = f"group_scale:{encoder_ffn_group_size}"
            else:
                scale_shape = [source.shape[0]]
                quant_kind = "quant"
                scale_kind = "scale"
            plan.append(OutputTensor(name, "I8", source.shape, quant_kind, (name,)))
            plan.append(
                OutputTensor(
                    name + "_int8_scale", "F32", scale_shape, scale_kind, (name,)
                )
            )
        elif is_decoder_proj(name):
            assert source.dtype == "BF16" and len(source.shape) == 2
            plan.append(OutputTensor(name, "I8", source.shape, "quant", (name,)))
            plan.append(
                OutputTensor(
                    name + "_int8_scale", "F32", [source.shape[0]], "scale", (name,)
                )
            )
        elif is_decoder_gate_proj(name):
            up_name = name.replace("gate_proj", "up_proj")
            up = sources[up_name]
            assert source.dtype == "BF16" and source.shape == up.shape
            fused_name = name.replace("gate_proj.weight", "gate_up_fused.weight")
            plan.append(
                OutputTensor(
                    fused_name,
                    "I8",
                    [2 * source.shape[0], source.shape[1]],
                    "fused_quant",
                    (name, up_name),
                )
            )
            plan.append(
                OutputTensor(
                    fused_name + "_int8_scale",
                    "F32",
                    [2 * source.shape[0]],
                    "fused_scale",
                    (name, up_name),
                )
            )
        elif is_decoder_up_proj(name):
            continue
        elif name == LM_HEAD:
            assert source.dtype == "BF16" and len(source.shape) == 2
            plan.append(OutputTensor(LM_HEAD, "I8", source.shape, "quant", (name,)))
            plan.append(OutputTensor(LM_HEAD_SCALE, "F32", [source.shape[0]], "scale", (name,)))
        elif name == EMBED_TOKENS and drop_embed_tokens:
            assert source.dtype == "BF16" and len(source.shape) == 2
            if not has_lm_head:
                plan.append(OutputTensor(LM_HEAD, "I8", source.shape, "quant", (name,)))
                plan.append(
                    OutputTensor(LM_HEAD_SCALE, "F32", [source.shape[0]], "scale", (name,))
                )
        else:
            plan.append(OutputTensor(name, source.dtype, source.shape, "copy", (name,)))
            if name == EMBED_TOKENS and not has_lm_head:
                assert source.dtype == "BF16" and len(source.shape) == 2
                plan.append(OutputTensor(LM_HEAD, "I8", source.shape, "quant", (name,)))
                plan.append(
                    OutputTensor(LM_HEAD_SCALE, "F32", [source.shape[0]], "scale", (name,))
                )
    return plan


def mmap_tensor(source: SourceTensor, dtype) -> np.ndarray:
    return np.memmap(
        source.path,
        dtype=dtype,
        mode="r",
        offset=source.data_start + source.start,
        shape=tuple(source.shape),
    )


def quantize_rows(
    source: SourceTensor,
    rows_per_chunk: int,
    adapter: LoRAProjection | None = None,
):
    assert source.dtype == "BF16" and len(source.shape) == 2
    values = mmap_tensor(source, np.uint16)
    rows, _ = source.shape
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        bf16 = np.asarray(values[row_start:row_end])
        f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
        if adapter is not None:
            if adapter.a.shape[1] != f32.shape[1] or adapter.b.shape[0] != rows:
                raise ValueError(
                    f"LoRA shape mismatch: base={source.shape} "
                    f"A={adapter.a.shape} B={adapter.b.shape}"
                )
            f32 = f32 + (adapter.b[row_start:row_end] @ adapter.a) * adapter.scale
        max_abs = np.max(np.abs(f32), axis=1)
        scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
        inverse = (127.0 / np.maximum(max_abs, 1e-10)).astype(np.float32)
        quantized = np.round(f32 * inverse[:, None]).clip(-127, 127).astype(np.int8)
        yield quantized, scales


def quantize_groups(
    source: SourceTensor,
    rows_per_chunk: int,
    group_size: int,
    adapter: LoRAProjection | None = None,
):
    assert source.dtype == "BF16" and len(source.shape) == 2
    values = mmap_tensor(source, np.uint16)
    rows, cols = source.shape
    assert cols % group_size == 0
    groups = cols // group_size
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        bf16 = np.asarray(values[row_start:row_end])
        f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
        if adapter is not None:
            if adapter.a.shape[1] != f32.shape[1] or adapter.b.shape[0] != rows:
                raise ValueError(
                    f"LoRA shape mismatch: base={source.shape} "
                    f"A={adapter.a.shape} B={adapter.b.shape}"
                )
            f32 = f32 + (adapter.b[row_start:row_end] @ adapter.a) * adapter.scale
        grouped = f32.reshape(row_end - row_start, groups, group_size)
        max_abs = np.max(np.abs(grouped), axis=2)
        scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
        inverse = (127.0 / np.maximum(max_abs, 1e-10)).astype(np.float32)
        quantized = np.round(grouped * inverse[:, :, None]).clip(-127, 127).astype(np.int8)
        yield quantized.reshape(row_end - row_start, cols), scales


def write_copy(stream, source: SourceTensor, chunk_bytes: int = 64 * 1024 * 1024):
    raw = np.memmap(
        source.path,
        dtype=np.uint8,
        mode="r",
        offset=source.data_start + source.start,
        shape=(source.end - source.start,),
    )
    for start in range(0, raw.size, chunk_bytes):
        stream.write(memoryview(raw[start : start + chunk_bytes]))


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def write_lora_merged_bf16(
    stream,
    source: SourceTensor,
    adapter: LoRAProjection,
    rows_per_chunk: int,
):
    assert source.dtype == "BF16" and len(source.shape) == 2
    values = mmap_tensor(source, np.uint16)
    rows, cols = source.shape
    if adapter.a.shape[1] != cols or adapter.b.shape[0] != rows:
        raise ValueError(
            f"LoRA shape mismatch: base={source.shape} "
            f"A={adapter.a.shape} B={adapter.b.shape}"
        )
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        bf16 = np.asarray(values[row_start:row_end])
        f32 = (bf16.astype(np.uint32) << 16).view(np.float32)
        effective = f32 + (adapter.b[row_start:row_end] @ adapter.a) * adapter.scale
        stream.write(memoryview(f32_to_bf16_bits(effective)).cast("B"))


def write_quant(
    stream,
    source: SourceTensor,
    rows_per_chunk: int,
    scales_only: bool,
    adapter: LoRAProjection | None = None,
):
    for quantized, scales in quantize_rows(source, rows_per_chunk, adapter):
        array = scales if scales_only else quantized
        stream.write(memoryview(np.ascontiguousarray(array)).cast("B"))


def write_group_quant(
    stream,
    source: SourceTensor,
    rows_per_chunk: int,
    group_size: int,
    scales_only: bool,
    adapter: LoRAProjection | None = None,
):
    for quantized, scales in quantize_groups(
        source, rows_per_chunk, group_size, adapter
    ):
        array = scales if scales_only else quantized
        stream.write(memoryview(np.ascontiguousarray(array)).cast("B"))


def write_fused(
    stream,
    gate: SourceTensor,
    up: SourceTensor,
    rows_per_chunk: int,
    scales_only: bool,
    gate_adapter: LoRAProjection | None = None,
    up_adapter: LoRAProjection | None = None,
):
    gate_chunks = quantize_rows(gate, rows_per_chunk, gate_adapter)
    up_chunks = quantize_rows(up, rows_per_chunk, up_adapter)
    for (gate_q, gate_s), (up_q, up_s) in zip(gate_chunks, up_chunks, strict=True):
        if scales_only:
            fused = np.empty(gate_s.size * 2, dtype=np.float32)
            fused[0::2] = gate_s
            fused[1::2] = up_s
        else:
            fused = np.empty((gate_q.shape[0] * 2, gate_q.shape[1]), dtype=np.int8)
            fused[0::2] = gate_q
            fused[1::2] = up_q
        stream.write(memoryview(fused).cast("B"))


def write_model(
    destination: Path,
    plan: list[OutputTensor],
    sources: dict[str, SourceTensor],
    rows_per_chunk: int,
    adapters: dict[str, LoRAProjection] | None = None,
):
    adapters = adapters or {}
    header = {}
    offset = 0
    for output in plan:
        header[output.name] = {
            "dtype": output.dtype,
            "shape": output.shape,
            "data_offsets": [offset, offset + output.size],
        }
        offset += output.size
    header_bytes = json.dumps(header, separators=(",", ":")).encode()

    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for index, output in enumerate(plan, 1):
            print(f"[{index:03d}/{len(plan):03d}] {output.kind:12s} {output.name}", flush=True)
            tensors = tuple(sources[name] for name in output.sources)
            if output.kind == "copy":
                write_copy(stream, tensors[0])
            elif output.kind == "copy_lora":
                write_lora_merged_bf16(
                    stream,
                    tensors[0],
                    adapters[output.sources[0]],
                    rows_per_chunk,
                )
            elif output.kind == "quant":
                write_quant(
                    stream,
                    tensors[0],
                    rows_per_chunk,
                    scales_only=False,
                    adapter=adapters.get(output.sources[0]),
                )
            elif output.kind == "scale":
                write_quant(
                    stream,
                    tensors[0],
                    rows_per_chunk,
                    scales_only=True,
                    adapter=adapters.get(output.sources[0]),
                )
            elif output.kind.startswith("group_quant:"):
                group_size = int(output.kind.split(":", 1)[1])
                write_group_quant(
                    stream,
                    tensors[0],
                    rows_per_chunk,
                    group_size,
                    scales_only=False,
                    adapter=adapters.get(output.sources[0]),
                )
            elif output.kind.startswith("group_scale:"):
                group_size = int(output.kind.split(":", 1)[1])
                write_group_quant(
                    stream,
                    tensors[0],
                    rows_per_chunk,
                    group_size,
                    scales_only=True,
                    adapter=adapters.get(output.sources[0]),
                )
            elif output.kind == "fused_quant":
                write_fused(
                    stream,
                    tensors[0],
                    tensors[1],
                    rows_per_chunk,
                    scales_only=False,
                    gate_adapter=adapters.get(output.sources[0]),
                    up_adapter=adapters.get(output.sources[1]),
                )
            elif output.kind == "fused_scale":
                write_fused(
                    stream,
                    tensors[0],
                    tensors[1],
                    rows_per_chunk,
                    scales_only=True,
                    gate_adapter=adapters.get(output.sources[0]),
                    up_adapter=adapters.get(output.sources[1]),
                )
            else:
                raise ValueError(output.kind)
        stream.flush()
        if stream.tell() != 8 + len(header_bytes) + offset:
            raise RuntimeError("output size does not match planned safetensors size")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lora-adapter-dir",
        help=(
            "merge an F32 LoRA adapter directly into each F32 row chunk before "
            "INT8 quantization, avoiding an intermediate BF16 merged checkpoint"
        ),
    )
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    parser.add_argument(
        "--drop-embed-tokens",
        action="store_true",
        help="reuse the tied rowwise-INT8 LM head for token lookup instead of retaining BF16 embeddings",
    )
    encoder_group = parser.add_mutually_exclusive_group()
    encoder_group.add_argument(
        "--int8-encoder-ffn",
        action="store_true",
        help="quantize audio-encoder FC1/FC2 weights to per-row INT8",
    )
    parser.add_argument(
        "--int8-encoder-attention",
        action="store_true",
        help="quantize encoder Q/K/V/output projection weights to per-row INT8",
    )
    parser.add_argument(
        "--encoder-attention-layer-start",
        type=int,
        help="first encoder layer whose attention projections are quantized (defaults to FFN start)",
    )
    parser.add_argument(
        "--encoder-attention-layer-limit",
        type=int,
        help="exclusive encoder attention layer limit (defaults to FFN limit)",
    )
    encoder_group.add_argument(
        "--int8-encoder-ffn-group-size",
        type=int,
        choices=(16, 32, 64, 128),
        help="quantize audio-encoder FC1/FC2 weights with symmetric INT8 groups",
    )
    parser.add_argument(
        "--encoder-ffn-layer-start",
        type=int,
        choices=range(0, 24),
        default=0,
        metavar="N",
        help="start encoder FFN quantization at layer N; default: 0",
    )
    parser.add_argument(
        "--encoder-ffn-layer-limit",
        type=int,
        choices=range(1, 25),
        metavar="N",
        help="quantize only encoder FFN layers [0, N); default: all 24",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    names, sources = load_sources(source_dir)
    adapters = (
        load_lora_adapter(Path(args.lora_adapter_dir))
        if args.lora_adapter_dir
        else {}
    )
    plan = build_plan(
        names,
        sources,
        drop_embed_tokens=args.drop_embed_tokens,
        int8_encoder_ffn=bool(
            args.int8_encoder_ffn or args.int8_encoder_ffn_group_size
        ),
        int8_encoder_attention=args.int8_encoder_attention,
        encoder_ffn_group_size=args.int8_encoder_ffn_group_size,
        encoder_ffn_layer_start=args.encoder_ffn_layer_start,
        encoder_ffn_layer_limit=args.encoder_ffn_layer_limit,
        encoder_attention_layer_start=args.encoder_attention_layer_start,
        encoder_attention_layer_limit=args.encoder_attention_layer_limit,
    )
    plan = [
        replace(output, kind="copy_lora")
        if output.kind == "copy" and output.sources[0] in adapters
        else output
        for output in plan
    ]
    handled_sources = {
        name
        for output in plan
        if output.kind != "copy"
        for name in output.sources
    }
    unmerged = sorted(set(adapters) - handled_sources)
    missing = sorted(set(adapters) - set(sources))
    if missing:
        raise ValueError(f"adapter targets missing from base model: {missing[:5]}")
    if unmerged:
        raise ValueError(
            f"adapter targets would remain unmerged: {unmerged[:5]}"
        )

    source_bytes = sum(source.end - source.start for source in sources.values())
    output_bytes = sum(output.size for output in plan)
    print(f"Source tensors: {len(sources)}, {source_bytes / 2**30:.2f} GiB")
    print(f"Output tensors: {len(plan)}, {output_bytes / 2**30:.2f} GiB")
    print(f"Projected reduction: {(1 - output_bytes / source_bytes) * 100:.1f}%")
    if args.plan_only:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_model(
        output_dir / "model.safetensors",
        plan,
        sources,
        args.rows_per_chunk,
        adapters,
    )
    for source in source_dir.iterdir():
        if source.is_file() and not source.name.endswith(".safetensors") and source.name != "model.safetensors.index.json":
            shutil.copy2(source, output_dir / source.name)
    if adapters:
        (output_dir / "int8_export_metadata.json").write_text(
            json.dumps(
                {
                    "lora_adapter_dir": str(Path(args.lora_adapter_dir).resolve()),
                    "lora_merge_dtype": "F32",
                    "intermediate_bf16_merge": False,
                    "quantizer": "symmetric_per_output_row_int8",
                    "range": [-127, 127],
                    "rounding": "nearest_even",
                    "adapter_targets": len(adapters),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(f"Wrote streaming INT8 model to {output_dir}")


if __name__ == "__main__":
    main()
