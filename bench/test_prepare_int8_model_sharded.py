import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
SPEC = importlib.util.spec_from_file_location(
    "prepare_int8_model_sharded", BENCH / "prepare_int8_model_sharded.py"
)
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    # Round-to-nearest-even, matching a normal F32 -> BF16 cast.
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


class DirectLoRAExportTest(unittest.TestCase):
    def test_adapter_name_maps_to_qwen_weight(self):
        self.assertEqual(
            exporter.adapter_target_to_base(
                "base_model.model.model.layers.3.self_attn.q_proj"
            ),
            "thinker.model.layers.3.self_attn.q_proj.weight",
        )
        self.assertEqual(
            exporter.adapter_target_to_base(
                "base_model.model.audio_tower.layers.7.fc1"
            ),
            "thinker.audio_tower.layers.7.fc1.weight",
        )

    def test_quantizes_f32_effective_weight_without_bf16_merge(self):
        base = np.array(
            [[0.5, -0.25, 0.125, -1.0], [0.0625, 0.75, -0.5, 0.25]],
            dtype=np.float32,
        )
        a = np.array([[0.03125, -0.021, 0.014, 0.008]], dtype=np.float32)
        b = np.array([[0.7], [-0.45]], dtype=np.float32)
        adapter = exporter.LoRAProjection(a=a, b=b, scale=0.5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.bf16"
            path.write_bytes(f32_to_bf16_bits(base).tobytes())
            source = exporter.SourceTensor(
                path=path,
                data_start=0,
                dtype="BF16",
                shape=list(base.shape),
                start=0,
                end=base.size * 2,
            )
            chunks = list(exporter.quantize_rows(source, rows_per_chunk=1, adapter=adapter))

        actual_q = np.concatenate([chunk[0] for chunk in chunks])
        actual_scales = np.concatenate([chunk[1] for chunk in chunks])
        bf16_base = (f32_to_bf16_bits(base).astype(np.uint32) << 16).view(np.float32)
        effective = bf16_base + (b @ a) * adapter.scale
        max_abs = np.max(np.abs(effective), axis=1)
        expected_scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
        expected_q = np.round(effective / expected_scales[:, None]).clip(-127, 127).astype(np.int8)
        np.testing.assert_array_equal(actual_q, expected_q)
        np.testing.assert_array_equal(actual_scales, expected_scales)

    def test_unquantized_adapter_target_is_merged_once_to_bf16(self):
        base = np.array([[0.5, -0.25], [0.125, -1.0]], dtype=np.float32)
        a = np.array([[0.03, -0.02]], dtype=np.float32)
        b = np.array([[0.7], [-0.45]], dtype=np.float32)
        adapter = exporter.LoRAProjection(a=a, b=b, scale=0.5)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "base.bf16"
            output_path = Path(directory) / "merged.bf16"
            source_path.write_bytes(f32_to_bf16_bits(base).tobytes())
            source = exporter.SourceTensor(
                path=source_path,
                data_start=0,
                dtype="BF16",
                shape=list(base.shape),
                start=0,
                end=base.size * 2,
            )
            with output_path.open("wb") as stream:
                exporter.write_lora_merged_bf16(stream, source, adapter, rows_per_chunk=1)
            actual = np.frombuffer(output_path.read_bytes(), dtype=np.uint16).reshape(base.shape)
        bf16_base = (f32_to_bf16_bits(base).astype(np.uint32) << 16).view(np.float32)
        expected = f32_to_bf16_bits(bf16_base + (b @ a) * adapter.scale)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
