use std::{
    fmt::Display,
    fs::File,
    io::{Read, Seek},
    marker::PhantomData,
};

use flate2::read::{DeflateDecoder, GzDecoder, ZlibDecoder};
use hpt_common::shape::shape::Shape;
use hpt_traits::tensor::{CommonBounds, TensorInfo};

use crate::{
    data_loader::{parse_header_compressed, TensorMeta},
    CPUTensorCreator, CompressionAlgo,
};

use super::save::Save;

pub trait Load: Sized + Save {
    fn load<P: Into<std::path::PathBuf>>(path: P) -> std::io::Result<Self>
    where
        <Self as Save>::Meta: MetaLoad<Output = Self>,
    {
        let path: std::path::PathBuf = path.into();
        let meta = parse_header_compressed::<Self, _>(&path).expect("failed to parse header");
        let mut file = File::open(path)?;
        meta.load(&mut file)
    }
}

pub(crate) fn load<'a, T: CommonBounds + bytemuck::AnyBitPattern, B: CPUTensorCreator>(
    file: &mut File,
    meta: &TensorMeta<T, B>,
) -> std::io::Result<<B as CPUTensorCreator>::Output>
where
    <B as CPUTensorCreator>::Output: Clone + TensorInfo + Display,
{
    if meta.dtype != T::STR {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "the dtype stored is {}, but the dtype requested is {}",
                meta.dtype,
                T::STR
            ),
        ));
    }
    // since the shape is scaled with mem_size, we need to scale it back
    let shape = meta.shape.clone();
    // create the tensor
    let tensor = B::empty(&shape)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

    if tensor.size() == 0 {
        return Ok(tensor);
    }
    #[allow(unused_mut)]
    let mut res: &mut [T] =
        unsafe { std::slice::from_raw_parts_mut(tensor.ptr().ptr, tensor.size()) };

    let mut res_idx = 0;
    for (_, idx, compressed_len, current_block_mem_size) in meta.indices.iter() {
        let uncompressed_data = uncompress_data(
            file,
            *idx as u64,
            *compressed_len,
            *current_block_mem_size,
            meta.compression_algo,
        )?;
        for idx in (0..uncompressed_data.len()).step_by(std::mem::size_of::<T>()) {
            let val = &uncompressed_data[idx..idx + std::mem::size_of::<T>()];
            res[res_idx] = *bytemuck::from_bytes(val);
            res_idx += 1;
        }
    }
    Ok(tensor)
}

fn uncompress_data(
    file: &mut File,
    idx: u64,
    compressed_len: usize,
    block_mem_size: usize,
    compression_type: CompressionAlgo,
) -> std::io::Result<Vec<u8>> {
    file.seek(std::io::SeekFrom::Start(idx))?;
    let mut compressed_vec = vec![0u8; compressed_len];
    file.read_exact(&mut compressed_vec)?;
    let mut uncompressed_data = vec![0u8; block_mem_size];
    match compression_type {
        CompressionAlgo::Gzip => {
            let mut decoder = GzDecoder::new(compressed_vec.as_slice());
            decoder.read_exact(&mut uncompressed_data)?;
        }
        CompressionAlgo::Deflate => {
            let mut decoder = DeflateDecoder::new(compressed_vec.as_slice());
            decoder.read_exact(&mut uncompressed_data)?;
        }
        CompressionAlgo::Zlib => {
            let mut decoder = ZlibDecoder::new(compressed_vec.as_slice());
            decoder.read_exact(&mut uncompressed_data)?;
        }
        CompressionAlgo::NoCompression => {
            uncompressed_data = compressed_vec;
        }
    }
    Ok(uncompressed_data)
}

pub trait MetaLoad: Sized {
    type Output;
    fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output>;
}

macro_rules! impl_load {
    ($struct:ty, $default:expr $(, $generics:tt)*) => {
        impl<$($generics)*> MetaLoad for $struct {
            type Output = $struct;
            fn load(&self, _: &mut std::fs::File) -> std::io::Result<Self::Output> {
                Ok(self.clone())
            }
        }
    };
}

impl_load!(bool, false);
impl_load!(u8, 0);
impl_load!(i8, 0);
impl_load!(u16, 0);
impl_load!(i16, 0);
impl_load!(u32, 0);
impl_load!(i32, 0);
impl_load!(u64, 0);
impl_load!(i64, 0);
impl_load!(f32, 0.0);
impl_load!(f64, 0.0);
impl_load!(usize, 0);
impl_load!(isize, 0);
impl_load!(half::f16, half::f16::ZERO);
impl_load!(half::bf16, half::bf16::ZERO);
impl_load!(String, String::new());
impl_load!(Shape, Shape::new([]));
impl_load!(PhantomData<T>, Self, T);

impl<T: MetaLoad> MetaLoad for Option<T> {
    type Output = Option<T::Output>;
    fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output> {
        match self {
            Some(x) => Ok(Some(T::load(x, file)?)),
            None => Ok(None),
        }
    }
}

impl<T: MetaLoad> MetaLoad for Vec<T> {
    type Output = Vec<T::Output>;
    fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output> {
        let mut res = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            res.push(T::load(&self[i], file)?);
        }
        Ok(res)
    }
}

impl<T: MetaLoad, const N: usize> MetaLoad for [T; N] {
    type Output = [T::Output; N];
    fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output> {
        let mut arr: [std::mem::MaybeUninit<T::Output>; N] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N {
            arr[i] = std::mem::MaybeUninit::new(T::load(&self[i], file)?);
        }
        Ok(unsafe {
            let ptr = &arr as *const _ as *const [T::Output; N];
            ptr.read()
        })
    }
}
