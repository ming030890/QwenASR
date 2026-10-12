use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DType {
    F32,
    I8,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct TensorView<'a> {
    pub(crate) dtype: DType,
    pub(crate) shape: &'a [usize],
    pub(crate) data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub(crate) fn f32_vec(self, name: &str) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            bail!("tensor '{name}' has dtype {:?}, expected F32", self.dtype);
        }
        if !self.data.len().is_multiple_of(4) {
            bail!("tensor '{name}' has a non-F32-aligned byte length");
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    pub(crate) fn i8_vec(self, name: &str) -> Result<Vec<i8>> {
        if self.dtype != DType::I8 {
            bail!("tensor '{name}' has dtype {:?}, expected I8", self.dtype);
        }
        Ok(self.data.iter().map(|value| *value as i8).collect())
    }
}

pub(crate) struct SafeTensors<'a> {
    tensors: HashMap<String, (DType, Vec<usize>, &'a [u8])>,
}

impl<'a> SafeTensors<'a> {
    pub(crate) fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            bail!("safetensors file is shorter than its header-size prefix");
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| anyhow!("safetensors header length overflows usize"))?;
        if header_end > bytes.len() {
            bail!("safetensors header extends past the file");
        }
        let header_text = std::str::from_utf8(&bytes[8..header_end])?;
        let header: HashMap<String, Value> = serde_json::from_str(header_text)?;
        let data = &bytes[header_end..];
        let mut tensors = HashMap::with_capacity(header.len());

        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let metadata: TensorMetadata = serde_json::from_value(value)
                .map_err(|error| anyhow!("invalid metadata for tensor '{name}': {error}"))?;
            let dtype = match metadata.dtype.as_str() {
                "F32" => DType::F32,
                "I8" => DType::I8,
                other => bail!("tensor '{name}' uses unsupported dtype '{other}'"),
            };
            let [start, end] = metadata.data_offsets;
            if start > end || end > data.len() {
                bail!("tensor '{name}' has out-of-bounds data offsets");
            }
            let elements = metadata.shape.iter().try_fold(1usize, |acc, dim| {
                acc.checked_mul(*dim)
                    .ok_or_else(|| anyhow!("tensor '{name}' shape overflows usize"))
            })?;
            let bytes_per_element = if dtype == DType::F32 { 4 } else { 1 };
            if elements != (end - start) / bytes_per_element
                || (end - start) % bytes_per_element != 0
            {
                bail!("tensor '{name}' shape does not match its data length");
            }
            tensors.insert(name, (dtype, metadata.shape, &data[start..end]));
        }

        Ok(Self { tensors })
    }

    pub(crate) fn get(&self, name: &str) -> Result<TensorView<'_>> {
        let (dtype, shape, data) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("required tensor '{name}' is missing"))?;
        Ok(TensorView {
            dtype: *dtype,
            shape,
            data,
        })
    }
}
