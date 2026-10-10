//! Safetensors mmap reader with multi-shard support.

use std::collections::HashMap;
use std::os::unix::io::RawFd;
use std::path::Path;

const MAX_TENSORS: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I8,
    I32,
    I64,
    Bool,
    Unknown,
}

#[inline]
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 0i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let f_exp = (((127 - 15 + 1 + e) as u32) & 0xff) << 23;
            let f_mant = (m & 0x3ff) << 13;
            f32::from_bits((sign << 31) | f_exp | f_mant)
        }
    } else if exp == 31 {
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | (0xff << 23) | f_mant)
    } else {
        let f_exp = (exp + 127 - 15) << 23;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | f_exp | f_mant)
    }
}

impl Dtype {
    fn from_str(s: &str) -> Self {
        match s {
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I8" => Dtype::I8,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            "BOOL" => Dtype::Bool,
            _ => Dtype::Unknown,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<i64>,
    pub data_offset: usize,
    pub data_size: usize,
}

impl TensorMeta {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }
}

pub struct SafetensorsFile {
    _fd: RawFd,
    data: *mut u8,
    file_size: usize,
    header_size: usize,
    pub tensors: Vec<TensorMeta>,
    tensor_map: HashMap<String, usize>,
}

unsafe impl Send for SafetensorsFile {}
unsafe impl Sync for SafetensorsFile {}

impl SafetensorsFile {
    pub fn open(path: &str) -> Option<Self> {
        use libc::*;
        use std::ffi::CString;

        let c_path = CString::new(path).ok()?;
        let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
        if fd < 0 {
            return None;
        }

        let mut stat_buf = unsafe { std::mem::zeroed::<stat>() };
        if unsafe { fstat(fd, &mut stat_buf) } < 0 {
            unsafe {
                close(fd);
            }
            return None;
        }

        let file_size = stat_buf.st_size as usize;
        if file_size < 8 {
            unsafe {
                close(fd);
            }
            return None;
        }

        let data = unsafe {
            mmap(
                std::ptr::null_mut(),
                file_size,
                PROT_READ,
                MAP_PRIVATE,
                fd,
                0,
            )
        };
        // Keep fd open for mmap lifetime? Actually mmap doesn't need it.
        // But we close it since MAP_PRIVATE doesn't need fd after mmap.
        let raw_fd = fd;
        unsafe {
            close(fd);
        }

        if data == libc::MAP_FAILED {
            return None;
        }
        let data = data as *mut u8;

        // Read header size (first 8 bytes, little-endian u64)
        let header_size = unsafe {
            let mut buf = [0u8; 8];
            std::ptr::copy_nonoverlapping(data, buf.as_mut_ptr(), 8);
            u64::from_le_bytes(buf) as usize
        };

        if header_size > file_size - 8 {
            unsafe {
                munmap(data as *mut _, file_size);
            }
            return None;
        }

        // Parse JSON header
        let header_json = unsafe {
            let slice = std::slice::from_raw_parts(data.add(8), header_size);
            std::str::from_utf8(slice).ok()?
        };

        let tensors = parse_header(header_json)?;
        let mut tensor_map = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            tensor_map.insert(t.name.clone(), i);
        }

        Some(SafetensorsFile {
            _fd: raw_fd,
            data,
            file_size,
            header_size,
            tensors,
            tensor_map,
        })
    }

    pub fn find(&self, name: &str) -> Option<&TensorMeta> {
        self.tensor_map.get(name).map(|&i| &self.tensors[i])
    }

    pub fn data_ptr(&self, tensor: &TensorMeta) -> *const u8 {
        unsafe { self.data.add(8 + self.header_size + tensor.data_offset) }
    }

    /// Get tensor data as f32 Vec (converts from BF16 if needed).
    pub fn get_f32(&self, tensor: &TensorMeta) -> Option<Vec<f32>> {
        let n = tensor.numel();
        if n == 0 {
            return None;
        }

        let ptr = self.data_ptr(tensor);
        match tensor.dtype {
            Dtype::F32 => {
                let mut out = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), n);
                }
                Some(out)
            }
            Dtype::BF16 => {
                let src = unsafe { std::slice::from_raw_parts(ptr as *const u16, n) };
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    out[i] = f32::from_bits((src[i] as u32) << 16);
                }
                Some(out)
            }
            Dtype::F16 => {
                let src = unsafe { std::slice::from_raw_parts(ptr as *const u16, n) };
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    out[i] = f16_to_f32(src[i]);
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// Get direct pointer to BF16 data in mmap.
    pub fn get_bf16_direct(&self, tensor: &TensorMeta) -> Option<*const u16> {
        if tensor.dtype != Dtype::BF16 {
            return None;
        }
        Some(self.data_ptr(tensor) as *const u16)
    }

    /// Copy I8 tensor data into a Vec<i8>.
    pub fn get_i8_vec(&self, tensor: &TensorMeta) -> Option<Vec<i8>> {
        if tensor.dtype != Dtype::I8 {
            return None;
        }
        let n = tensor.numel();
        if n == 0 {
            return None;
        }
        let ptr = self.data_ptr(tensor) as *const i8;
        let mut out = vec![0i8; n];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        Some(out)
    }

    /// Copy an I32 tensor into an owned vector.
    pub fn get_i32_vec(&self, tensor: &TensorMeta) -> Option<Vec<i32>> {
        if tensor.dtype != Dtype::I32 {
            return None;
        }
        let n = tensor.numel();
        if n == 0 {
            return None;
        }
        let ptr = self.data_ptr(tensor) as *const i32;
        let mut out = vec![0i32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), n);
        }
        Some(out)
    }

    /// Get direct zero-copy pointer to I8 data in the mmap region.
    pub fn get_i8_ptr(&self, tensor: &TensorMeta) -> Option<(*const i8, usize)> {
        if tensor.dtype != Dtype::I8 {
            return None;
        }
        let n = tensor.numel();
        if n == 0 {
            return None;
        }
        Some((self.data_ptr(tensor) as *const i8, n))
    }

    /// Get a zero-copy pointer to F32 tensor data in the mmap region.
    pub fn get_f32_ptr(&self, tensor: &TensorMeta) -> Option<(*const f32, usize)> {
        if tensor.dtype != Dtype::F32 {
            return None;
        }
        let n = tensor.numel();
        if n == 0 {
            return None;
        }
        Some((self.data_ptr(tensor) as *const f32, n))
    }
}

impl Drop for SafetensorsFile {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                libc::munmap(self.data as *mut _, self.file_size);
            }
        }
    }
}

pub struct MultiSafetensors {
    pub shards: Vec<SafetensorsFile>,
}

impl MultiSafetensors {
    pub fn open(model_dir: &str) -> Option<Self> {
        // Try single file first
        let single_path = format!("{}/model.safetensors", model_dir);
        if let Some(sf) = SafetensorsFile::open(&single_path) {
            let mut shards = vec![sf];
            let packed_path = format!("{}/model.kleidiai.safetensors", model_dir);
            if Path::new(&packed_path).is_file() {
                match SafetensorsFile::open(&packed_path) {
                    Some(packed) => shards.push(packed),
                    None => {
                        eprintln!("multi_safetensors_open: failed to open {}", packed_path);
                        return None;
                    }
                }
            }
            return Some(MultiSafetensors { shards });
        }

        // Scan directory for shard files
        let dir = Path::new(model_dir);
        let mut shard_names: Vec<String> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    shard_names.push(name);
                }
            }
        }

        if shard_names.is_empty() {
            eprintln!(
                "multi_safetensors_open: no safetensors files in {}",
                model_dir
            );
            return None;
        }

        shard_names.sort();

        let mut shards = Vec::new();
        for name in &shard_names {
            let path = format!("{}/{}", model_dir, name);
            match SafetensorsFile::open(&path) {
                Some(sf) => shards.push(sf),
                None => {
                    eprintln!("multi_safetensors_open: failed to open {}", path);
                    return None;
                }
            }
        }

        Some(MultiSafetensors { shards })
    }

    /// Advise kernel to preload Shard 1 (Encoder: 320 MB) into RAM
    pub fn preload_encoder(&self) {
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "macos"))]
        if let Some(shard) = self.shards.first() {
            unsafe {
                libc::madvise(shard.data as *mut _, shard.file_size, libc::MADV_WILLNEED);
            }
        }
    }

    /// Advise kernel to preload Shards 2..N (Decoder: 1.64 GB) into RAM
    pub fn preload_decoder(&self) {
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "macos"))]
        for shard in self.shards.iter().skip(1) {
            unsafe {
                libc::madvise(shard.data as *mut _, shard.file_size, libc::MADV_WILLNEED);
            }
        }
    }

    /// Find a tensor by name across all shards.
    /// Returns (shard_index, TensorMeta).
    pub fn find(&self, name: &str) -> Option<(usize, &TensorMeta)> {
        for (si, shard) in self.shards.iter().enumerate() {
            if let Some(t) = shard.find(name) {
                return Some((si, t));
            }
        }
        None
    }

    /// Convenience: get f32 data for a named tensor.
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        let (si, t) = self.find(name)?;
        self.shards[si].get_f32(t)
    }

    /// Convenience: get direct BF16 pointer for a named tensor.
    pub fn get_bf16_direct(&self, name: &str) -> Option<*const u16> {
        let (si, t) = self.find(name)?;
        self.shards[si].get_bf16_direct(t)
    }

    /// Convenience: copy I8 tensor data into a Vec<i8>.
    pub fn get_i8_vec(&self, name: &str) -> Option<Vec<i8>> {
        let (si, t) = self.find(name)?;
        self.shards[si].get_i8_vec(t)
    }

    /// Convenience: zero-copy pointer into I8 tensor in the mmap region.
    pub fn get_i8_ptr(&self, name: &str) -> Option<(*const i8, usize)> {
        let (si, t) = self.find(name)?;
        self.shards[si].get_i8_ptr(t)
    }

    /// Convenience: zero-copy pointer into an F32 tensor in the mmap region.
    pub fn get_f32_ptr(&self, name: &str) -> Option<(*const f32, usize)> {
        let (si, t) = self.find(name)?;
        self.shards[si].get_f32_ptr(t)
    }

    /// Return the dtype of a named tensor, or None if not found.
    pub fn get_dtype(&self, name: &str) -> Option<Dtype> {
        let (_, t) = self.find(name)?;
        Some(t.dtype)
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.find(name).is_some()
    }
}

// ========================================================================
// Minimal JSON parser for safetensors header
// ========================================================================

fn parse_header(json: &str) -> Option<Vec<TensorMeta>> {
    let mut tensors = Vec::new();
    let bytes = json.as_bytes();
    let mut pos = 0;

    skip_whitespace(bytes, &mut pos);
    if pos >= bytes.len() || bytes[pos] != b'{' {
        return None;
    }
    pos += 1;

    loop {
        skip_whitespace(bytes, &mut pos);
        if pos >= bytes.len() {
            break;
        }
        if bytes[pos] == b'}' {
            break;
        }
        if bytes[pos] == b',' {
            pos += 1;
            continue;
        }

        // Parse key
        let key = parse_json_string(bytes, &mut pos)?;
        skip_whitespace(bytes, &mut pos);
        if pos >= bytes.len() || bytes[pos] != b':' {
            return None;
        }
        pos += 1;

        if key == "__metadata__" {
            skip_json_value(bytes, &mut pos);
            continue;
        }

        // Parse tensor entry
        let tensor = parse_tensor_entry(bytes, &mut pos, &key)?;
        tensors.push(tensor);

        if tensors.len() >= MAX_TENSORS {
            break;
        }
    }

    Some(tensors)
}

fn skip_whitespace(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() {
        match bytes[*pos] {
            b' ' | b'\n' | b'\r' | b'\t' => *pos += 1,
            _ => break,
        }
    }
}

fn parse_json_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    skip_whitespace(bytes, pos);
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return None;
    }
    *pos += 1;

    let mut result = Vec::new();
    while *pos < bytes.len() && bytes[*pos] != b'"' {
        if bytes[*pos] == b'\\' {
            *pos += 1;
            if *pos >= bytes.len() {
                return None;
            }
            match bytes[*pos] {
                b'n' => result.push(b'\n'),
                b't' => result.push(b'\t'),
                b'"' => result.push(b'"'),
                b'\\' => result.push(b'\\'),
                b'/' => result.push(b'/'),
                b'u' => {
                    *pos += 1;
                    let mut cp = 0u32;
                    for _ in 0..4 {
                        if *pos >= bytes.len() {
                            return None;
                        }
                        cp <<= 4;
                        let c = bytes[*pos];
                        cp |= match c {
                            b'0'..=b'9' => (c - b'0') as u32,
                            b'a'..=b'f' => (c - b'a' + 10) as u32,
                            b'A'..=b'F' => (c - b'A' + 10) as u32,
                            _ => return None,
                        };
                        *pos += 1;
                    }
                    // Encode cp as UTF-8
                    let ch = char::from_u32(cp).unwrap_or('?');
                    let mut buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut buf);
                    result.extend_from_slice(s.as_bytes());
                    continue; // skip the pos += 1 below
                }
                other => result.push(other),
            }
        } else {
            result.push(bytes[*pos]);
        }
        *pos += 1;
    }

    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return None;
    }
    *pos += 1;

    String::from_utf8(result).ok()
}

fn parse_json_int(bytes: &[u8], pos: &mut usize) -> Option<i64> {
    skip_whitespace(bytes, pos);
    let mut neg = false;
    if *pos < bytes.len() && bytes[*pos] == b'-' {
        neg = true;
        *pos += 1;
    }
    let mut val: i64 = 0;
    let mut found = false;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        val = val * 10 + (bytes[*pos] - b'0') as i64;
        *pos += 1;
        found = true;
    }
    if !found {
        return None;
    }
    Some(if neg { -val } else { val })
}

fn parse_tensor_entry(bytes: &[u8], pos: &mut usize, name: &str) -> Option<TensorMeta> {
    skip_whitespace(bytes, pos);
    if *pos >= bytes.len() || bytes[*pos] != b'{' {
        return None;
    }
    *pos += 1;

    let mut dtype = Dtype::Unknown;
    let mut shape = Vec::new();
    let mut data_offset: usize = 0;
    let mut data_size: usize = 0;

    loop {
        skip_whitespace(bytes, pos);
        if *pos >= bytes.len() {
            break;
        }
        if bytes[*pos] == b'}' {
            *pos += 1;
            break;
        }
        if bytes[*pos] == b',' {
            *pos += 1;
            continue;
        }

        let key = parse_json_string(bytes, pos)?;
        skip_whitespace(bytes, pos);
        if *pos >= bytes.len() || bytes[*pos] != b':' {
            return None;
        }
        *pos += 1;
        skip_whitespace(bytes, pos);

        match key.as_str() {
            "dtype" => {
                let dtype_str = parse_json_string(bytes, pos)?;
                dtype = Dtype::from_str(&dtype_str);
            }
            "shape" => {
                if *pos >= bytes.len() || bytes[*pos] != b'[' {
                    return None;
                }
                *pos += 1;
                loop {
                    skip_whitespace(bytes, pos);
                    if *pos >= bytes.len() {
                        break;
                    }
                    if bytes[*pos] == b']' {
                        *pos += 1;
                        break;
                    }
                    if bytes[*pos] == b',' {
                        *pos += 1;
                        continue;
                    }
                    let dim = parse_json_int(bytes, pos)?;
                    shape.push(dim);
                }
            }
            "data_offsets" => {
                if *pos >= bytes.len() || bytes[*pos] != b'[' {
                    return None;
                }
                *pos += 1;
                skip_whitespace(bytes, pos);
                let start = parse_json_int(bytes, pos)? as usize;
                skip_whitespace(bytes, pos);
                if *pos < bytes.len() && bytes[*pos] == b',' {
                    *pos += 1;
                }
                skip_whitespace(bytes, pos);
                let end = parse_json_int(bytes, pos)? as usize;
                skip_whitespace(bytes, pos);
                if *pos < bytes.len() && bytes[*pos] == b']' {
                    *pos += 1;
                }
                data_offset = start;
                data_size = end - start;
            }
            _ => {
                skip_json_value(bytes, pos);
            }
        }
    }

    Some(TensorMeta {
        name: name.to_string(),
        dtype,
        shape,
        data_offset,
        data_size,
    })
}

fn skip_json_value(bytes: &[u8], pos: &mut usize) {
    skip_whitespace(bytes, pos);
    if *pos >= bytes.len() {
        return;
    }

    match bytes[*pos] {
        b'"' => {
            *pos += 1;
            while *pos < bytes.len() && bytes[*pos] != b'"' {
                if bytes[*pos] == b'\\' {
                    *pos += 1;
                }
                if *pos < bytes.len() {
                    *pos += 1;
                }
            }
            if *pos < bytes.len() {
                *pos += 1;
            }
        }
        b'[' => {
            let mut depth = 1;
            *pos += 1;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    b'"' => {
                        *pos += 1;
                        while *pos < bytes.len() && bytes[*pos] != b'"' {
                            if bytes[*pos] == b'\\' {
                                *pos += 1;
                            }
                            if *pos < bytes.len() {
                                *pos += 1;
                            }
                        }
                    }
                    _ => {}
                }
                *pos += 1;
            }
        }
        b'{' => {
            let mut depth = 1;
            *pos += 1;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    b'"' => {
                        *pos += 1;
                        while *pos < bytes.len() && bytes[*pos] != b'"' {
                            if bytes[*pos] == b'\\' {
                                *pos += 1;
                            }
                            if *pos < bytes.len() {
                                *pos += 1;
                            }
                        }
                    }
                    _ => {}
                }
                *pos += 1;
            }
        }
        _ => {
            // Number, bool, null
            while *pos < bytes.len()
                && bytes[*pos] != b','
                && bytes[*pos] != b'}'
                && bytes[*pos] != b']'
            {
                *pos += 1;
            }
        }
    }
}
