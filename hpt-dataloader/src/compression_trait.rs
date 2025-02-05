use std::{collections::HashMap, io::Write};

use hpt_common::{shape::shape::Shape, slice::Slice, strides::strides::Strides};
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo};
use hpt_types::dtype::Dtype;
use num::traits::{FromBytes, ToBytes};
use serde::{Deserialize, Serialize};

use crate::{
    data_loader::{Endian, HeaderInfo},
    load::load_compressed_slice,
    save::save,
};

pub trait CompressionTrait {
    fn write_all_data(&mut self, buf: &[u8]) -> std::io::Result<()>;
    fn flush_all(&mut self) -> std::io::Result<()>;
    fn finish_all(self) -> std::io::Result<Vec<u8>>;
}

macro_rules! impl_compression_trait {
    ($([$name:expr, $($t:ty),*]),*) => {
        $(
            impl CompressionTrait for $($t)* {
                fn write_all_data(&mut self, buf: &[u8]) -> std::io::Result<()> {
                    self.write_all(buf)
                }
                fn flush_all(&mut self) -> std::io::Result<()> {
                    self.flush()
                }
                fn finish_all(self) -> std::io::Result<Vec<u8>> {
                    self.finish()
                }
            }
        )*
    };
}

impl_compression_trait!(
    ["gzip", flate2::write::GzEncoder<Vec<u8>>],
    ["deflate", flate2::write::DeflateEncoder<Vec<u8>>],
    ["zlib", flate2::write::ZlibEncoder<Vec<u8>>]
);

pub trait DataLoaderTrait {
    fn shape(&self) -> &Shape;
    fn strides(&self) -> &Strides;
    fn fill_ne_bytes_slice(&self, offset: isize, writer: &mut [u8]);
    fn fill_be_bytes_slice(&self, offset: isize, writer: &mut [u8]);
    fn fill_le_bytes_slice(&self, offset: isize, writer: &mut [u8]);
    fn offset(&mut self, offset: isize);
    fn size(&self) -> usize;
    fn dtype(&self) -> Dtype;
    fn mem_size(&self) -> usize;
}

pub struct DataLoader<T> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) data: *const T,
}

impl<T> DataLoader<T> {
    pub fn new(shape: Shape, strides: Strides, data: *const T) -> Self {
        Self {
            shape,
            strides,
            data,
        }
    }
}

impl<T, const N: usize> DataLoaderTrait for DataLoader<T>
where
    T: CommonBounds + ToBytes<Bytes = [u8; N]>,
{
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &Strides {
        &self.strides
    }

    fn offset(&mut self, offset: isize) {
        self.data = unsafe { self.data.offset(offset) };
    }

    fn size(&self) -> usize {
        self.shape.size() as usize
    }

    fn mem_size(&self) -> usize {
        std::mem::size_of::<T>()
    }

    fn fill_ne_bytes_slice(&self, offset: isize, writer: &mut [u8]) {
        let val = unsafe { self.data.offset(offset).read() };
        writer.copy_from_slice(&val.to_ne_bytes());
    }

    fn fill_be_bytes_slice(&self, offset: isize, writer: &mut [u8]) {
        let val = unsafe { self.data.offset(offset).read() };
        writer.copy_from_slice(&val.to_be_bytes());
    }

    fn fill_le_bytes_slice(&self, offset: isize, writer: &mut [u8]) {
        let val = unsafe { self.data.offset(offset).read() };
        writer.copy_from_slice(&val.to_le_bytes());
    }

    fn dtype(&self) -> Dtype {
        T::ID
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgo {
    Gzip,
    Deflate,
    Zlib,
    NoCompression,
}

pub struct Meta {
    pub name: String,
    pub compression_algo: CompressionAlgo,
    pub endian: Endian,
    pub data_saver: Box<dyn DataLoaderTrait>,
    pub compression_level: u32,
}

pub struct TensorSaver {
    file_path: std::path::PathBuf,
    to_saves: Option<Vec<Meta>>,
}

impl TensorSaver {
    pub fn new(file_path: std::path::PathBuf) -> Self {
        Self {
            file_path,
            to_saves: None,
        }
    }

    pub fn push<
        const N: usize,
        T: ToBytes<Bytes = [u8; N]> + CommonBounds,
        A: Into<DataLoader<T>>,
    >(
        mut self,
        name: &str,
        tensor: A,
        compression_algo: CompressionAlgo,
        endian: Endian,
        compression_level: u32,
    ) -> Self {
        let data_loader = tensor.into();
        let meta = Meta {
            name: name.to_string(),
            compression_algo,
            endian,
            data_saver: Box::new(data_loader),
            compression_level,
        };
        if let Some(to_saves) = &mut self.to_saves {
            to_saves.push(meta);
        } else {
            self.to_saves = Some(vec![meta]);
        }
        self
    }

    pub fn save(self) -> std::io::Result<()> {
        save(
            self.file_path.to_str().unwrap().into(),
            self.to_saves.unwrap(),
        )
    }
}

pub struct TensorLoader {
    file_path: std::path::PathBuf,
    to_loads: Option<Vec<(String, Vec<Slice>)>>,
}

impl TensorLoader {
    pub fn new(file_path: std::path::PathBuf) -> Self {
        Self {
            file_path,
            to_loads: None,
        }
    }

    pub fn push(mut self, name: &str, slices: &[Slice]) -> Self {
        if let Some(to_loads) = &mut self.to_loads {
            to_loads.push((name.to_string(), slices.to_vec()));
        } else {
            self.to_loads = Some(vec![(name.to_string(), slices.to_vec())]);
        }
        self
    }

    pub fn load<T, B, const N: usize>(self) -> std::io::Result<HashMap<String, B>>
    where
        T: CommonBounds + FromBytes<Bytes = [u8; N]>,
        B: TensorCreator<T, Output = B> + Clone + TensorInfo<T>,
    {
        let res = load_compressed_slice::<T, B, N>(
            self.file_path.to_str().unwrap().into(),
            self.to_loads.expect("no tensors to load"),
        )
        .expect("failed to load tensor");
        Ok(res)
    }

    pub fn load_all<T, B, const N: usize>(self) -> std::io::Result<HashMap<String, B>>
    where
        T: CommonBounds + FromBytes<Bytes = [u8; N]>,
        B: TensorCreator<T, Output = B> + Clone + TensorInfo<T>,
    {
        let res = HeaderInfo::parse_header_compressed(self.file_path.to_str().unwrap().into())
            .expect("failed to parse header");
        let res = load_compressed_slice::<T, B, N>(
            self.file_path.to_str().unwrap().into(),
            res.into_values().map(|x| (x.name, vec![])).collect(),
        )
        .expect("failed to load tensor");
        Ok(res)
    }
}
