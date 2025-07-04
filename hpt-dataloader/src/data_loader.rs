use hpt_traits::tensor::{CommonBounds, TensorInfo};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::marker::PhantomData;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek},
};

use crate::struct_save::load::load;
use crate::struct_save::load::MetaLoad;
use crate::struct_save::save::Save;
use crate::CPUTensorCreator;
use crate::CompressionAlgo;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum Endian {
    Little,
    Big,
    Native,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct HeaderInfos {
    pub(crate) begin: u64,
    pub(crate) infos: Vec<HeaderInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct HeaderInfo {
    pub(crate) begin: u64,
    pub(crate) name: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) size: usize,
    pub(crate) indices: Vec<(usize, usize, usize, usize)>,
    pub(crate) compress_algo: CompressionAlgo,
    pub(crate) dtype: String,
    pub(crate) sizeof: usize,
    pub(crate) endian: Endian,
}
/// the meta data of the tensor
#[derive(Serialize, Deserialize)]
#[must_use]
pub struct TensorMeta<T: CommonBounds, B: CPUTensorCreator>
where
    <B as CPUTensorCreator>::Output: Clone + TensorInfo
{
    pub begin: usize,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub size: usize,
    pub dtype: String,
    pub compression_algo: CompressionAlgo,
    pub endian: Endian,
    pub indices: Vec<(usize, usize, usize, usize)>,
    pub phantom: PhantomData<(T, B)>,
}

pub fn parse_header_compressed<M: Save, P: Into<std::path::PathBuf>>(
    file: P,
) -> Result<<M as Save>::Meta, Box<dyn std::error::Error>> {
    let mut file = File::open(file.into())?;
    file.read_exact(&mut [0u8; "FASTTENSOR".len()])?;
    let mut header_infos = [0u8; 20];
    file.read_exact(&mut header_infos)?;
    let header = std::str::from_utf8(&header_infos)?;
    let header_int = header.trim().parse::<u64>()?;
    file.seek(std::io::SeekFrom::Start(header_int))?; // offset for header
    let mut buffer3 = vec![];
    file.read_to_end(&mut buffer3)?;
    let info = std::str::from_utf8(&buffer3)?;
    let ret = serde_json::from_str::<M::Meta>(info)?;
    Ok(ret)
}

impl<T, B: CPUTensorCreator> MetaLoad for TensorMeta<T, B>
where
    T: CommonBounds + bytemuck::AnyBitPattern,
    <B as CPUTensorCreator>::Output: Clone + TensorInfo + Display + Into<B>,
{
    type Output = B;
    fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output> {
        Ok(load::<T, B>(file, self)?.into())
    }
}

impl HeaderInfo {
    pub(crate) fn parse_header_compressed(
        file: &str,
    ) -> Result<HashMap<String, HeaderInfo>, Box<dyn std::error::Error>> {
        let mut file = File::open(file)?;
        file.read_exact(&mut [0u8; "FASTTENSOR".len()])?;
        let mut header_infos = [0u8; 20];
        file.read_exact(&mut header_infos)?;
        let header = std::str::from_utf8(&header_infos)?;
        let header_int = header.trim().parse::<u64>()?;
        file.seek(std::io::SeekFrom::Start(header_int))?; // offset for header
        let mut buffer3 = vec![];
        file.read_to_end(&mut buffer3)?;
        let info = std::str::from_utf8(&buffer3)?;
        let ret: HashMap<String, HeaderInfo> =
            serde_json::from_str::<HashMap<String, Self>>(&info)?;
        Ok(ret)
    }
}

pub struct LoadResult {
    pub(crate) ptr: *mut u8,
    pub(crate) shape: Vec<i64>,
    pub(crate) strides: Vec<i64>,
    pub(crate) dtype: String
}

impl LoadResult {
    pub fn dtype(&self) -> &str {
        &self.dtype
    }
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn size(&self) -> usize {
        self.shape.iter().map(|&x| x as usize).product()
    }
}

impl Drop for LoadResult {
    fn drop(&mut self) {
        unsafe {
            let layout = std::alloc::Layout::from_size_align(
                self.shape.iter().product::<i64>() as usize, 
                128
            ).unwrap();
            std::alloc::dealloc(self.ptr, layout);
        }
    }
}