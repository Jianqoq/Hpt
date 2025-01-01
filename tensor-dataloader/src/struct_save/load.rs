use std::{
    fs::File,
    io::{Read, Seek},
};

use flate2::read::{DeflateDecoder, GzDecoder, ZlibDecoder};
use num::traits::FromBytes;
use tensor_traits::{CommonBounds, TensorCreator, TensorInfo};

use crate::{
    data_loader::{parse_header_compressed, TensorMeta},
    CompressionAlgo, Endian,
};

use super::save::Save;

pub trait Load: Sized + Save {
    fn load(path: &str) -> std::io::Result<Self>
    where
        <Self as Save>::Meta: MetaLoad<Output = Self>,
    {
        let meta = parse_header_compressed::<Self>(path).expect("failed to parse header");
        let mut file = File::open(path)?;
        meta.load(&mut file)
    }
}

pub(crate) fn load<
    'a,
    T: CommonBounds + FromBytes<Bytes = [u8; N]>,
    B: TensorCreator<T, Output = B> + Clone + TensorInfo<T>,
    const N: usize,
>(
    file: &mut File,
    meta: &TensorMeta<T, B>,
) -> std::io::Result<B> {
    if meta.dtype != T::ID {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "the dtype stored is {}, but the dtype requested is {}",
                meta.dtype,
                T::ID
            ),
        ));
    }
    // since the shape is scaled with mem_size, we need to scale it back
    let shape = meta.shape.clone();
    // create the tensor
    let tensor: B = B::empty(&shape)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;

    if tensor.size() == 0 {
        return Ok(tensor);
    }
    #[allow(unused_mut)]
    let mut res: &mut [T] =
        unsafe { std::slice::from_raw_parts_mut(tensor.ptr().ptr, tensor.size()) };

    let pack = get_pack_closure::<T, N>(meta.endian);

    for (_, idx, compressed_len, current_block_mem_size) in meta.indices.iter() {
        let uncompressed_data = uncompress_data(
            file,
            *idx as u64,
            *compressed_len,
            *current_block_mem_size,
            meta.compression_algo,
        )?;
        for (i, idx) in (0..uncompressed_data.len())
            .step_by(std::mem::size_of::<T>())
            .enumerate()
        {
            let val = &uncompressed_data[idx..idx + std::mem::size_of::<T>()];
            res[i] = pack(val);
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

fn get_pack_closure<T: CommonBounds + FromBytes<Bytes = [u8; N]>, const N: usize>(
    endian: Endian,
) -> impl Fn(&[u8]) -> T {
    match endian {
        Endian::Little => |val: &[u8]| T::from_le_bytes(val.try_into().unwrap()),
        Endian::Big => |val: &[u8]| T::from_be_bytes(val.try_into().unwrap()),
        Endian::Native => |val: &[u8]| T::from_ne_bytes(val.try_into().unwrap()),
    }
}

pub trait MetaLoad: Sized {
    type Output;
    fn load(&self, file: &mut std::fs::File) -> std::io::Result<Self::Output>;
}

macro_rules! impl_load {
    ($struct:ident) => {
        impl MetaLoad for $struct {
            type Output = $struct;
            fn load(&self, _: &mut std::fs::File) -> std::io::Result<Self::Output> {
                Ok(*self)
            }
        }
    };
}

impl_load!(bool);
impl_load!(u8);
impl_load!(i8);
impl_load!(u16);
impl_load!(i16);
impl_load!(u32);
impl_load!(i32);
impl_load!(u64);
impl_load!(i64);
impl_load!(f32);
impl_load!(f64);
