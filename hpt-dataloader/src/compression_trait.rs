use std::{ collections::HashMap, io::Write };

use hpt_common::{ shape::shape::Shape, strides::strides::Strides };
use hpt_traits::tensor::{ CommonBounds, TensorInfo };
use hpt_types::dtype::TypeCommon;
use serde::{ Deserialize, Serialize };

use crate::{
    data_loader::{ Endian, LoadResult },
    load::load_compressed_slice,
    save::save,
    utils::ToDataLoader,
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
    fn dtype(&self) -> &'static str;
    fn mem_size(&self) -> usize;
}

pub struct DataLoader<T, B> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    #[allow(dead_code)]
    pub(crate) tensor: B, // this is just used to let rust hold the data not to drop
    pub(crate) data: *const T,
}

impl<T, B> DataLoader<T, B> where B: TensorInfo {
    pub fn new(shape: Shape, strides: Strides, tensor: B) -> Self {
        let ptr = tensor.ptr();
        Self {
            shape,
            strides,
            tensor,
            data: ptr.ptr as *const T,
        }
    }
}

impl<B, T> DataLoaderTrait
    for DataLoader<T, B>
    where B: TensorInfo, T: CommonBounds + bytemuck::NoUninit
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
        writer.copy_from_slice(bytemuck::bytes_of(&val));
    }

    fn fill_be_bytes_slice(&self, offset: isize, writer: &mut [u8]) {
        let val = unsafe { self.data.offset(offset).read() };
        writer.copy_from_slice(bytemuck::bytes_of(&val));
    }

    fn fill_le_bytes_slice(&self, offset: isize, writer: &mut [u8]) {
        let val = unsafe { self.data.offset(offset).read() };
        writer.copy_from_slice(bytemuck::bytes_of(&val));
    }

    fn dtype(&self) -> &'static str {
        T::STR
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
    pub fn new<P: Into<std::path::PathBuf>>(file_path: P) -> Self {
        Self {
            file_path: file_path.into(),
            to_saves: None,
        }
    }

    pub fn push<A>(
        mut self,
        name: &str,
        tensor: A,
        compression_algo: CompressionAlgo,
        compression_level: u32
    )
        -> Self
        where A: TensorInfo + 'static + ToDataLoader, <A as ToDataLoader>::Output: DataLoaderTrait
    {
        let data_loader = tensor.to_dataloader();
        let meta = Meta {
            name: name.to_string(),
            compression_algo,
            endian: Endian::Native,
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
        save(self.file_path.to_str().unwrap().into(), self.to_saves.unwrap())
    }
}

pub struct TensorLoader {
    file_path: std::path::PathBuf,
    to_loads: Option<Vec<(String, Vec<(i64, i64, i64)>, String, usize)>>,
}

impl TensorLoader {
    pub fn new<P: Into<std::path::PathBuf>>(file_path: P) -> Self {
        Self {
            file_path: file_path.into(),
            to_loads: None,
        }
    }

    pub fn push<T: TypeCommon>(mut self, name: &str, slices: &[(i64, i64, i64)]) -> Self {
        if let Some(to_loads) = &mut self.to_loads {
            to_loads.push((
                name.to_string(),
                slices.to_vec(),
                T::STR.to_string(),
                std::mem::size_of::<T>(),
            ));
        } else {
            self.to_loads = Some(
                vec![(
                    name.to_string(),
                    slices.to_vec(),
                    T::STR.to_string(),
                    std::mem::size_of::<T>(),
                )]
            );
        }
        self
    }

    pub fn load(self) -> std::io::Result<HashMap<String, LoadResult>> {
        let res = load_compressed_slice(
            self.file_path.to_str().unwrap().into(),
            self.to_loads.expect("no tensors to load")
        ).expect("failed to load tensor");
        Ok(res)
    }
}
