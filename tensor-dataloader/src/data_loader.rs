use anyhow::Ok;
use anyhow::Result;
use indicatif::ProgressBar;
use num::traits::FromBytes;
use num::traits::ToBytes;
use serde::{ Deserialize, Serialize };
use tensor_common::pointer::Pointer;
use tensor_common::slice::slice_process;
use tensor_common::slice::Slice;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use std::io::BufWriter;
use std::{ collections::HashMap, fs::File, io::{ Read, Seek, Write } };
use crate::compression_trait::CompressionTrait;

const CHUNK_BUFF: usize = 1024 * 1024;

/// count the number of arguments in pass to the macro
#[macro_export]
macro_rules! count_args {
    (
        $name:expr => [$($indexes:tt)*]
    ) => {
        1
    };
    (
        $name:expr => [$($indexes:tt)*],
        $(
            $name2:expr => [$($indexes2:tt)*]
        ),*
    ) => {
        1 + count_args!($($name2 => [$($indexes2)*]),*)
    };
    ($name:ident) => {
        1
    };
    ($name:ident, $($next:ident)*) => {
        1 + count_args!($($next)*)
    };
    ([$name:expr, $tensors:expr], $([$name2:expr, $tensors2:expr]),*) => {
        1 + count_args!($([$name2, $tensors2], )*)
    };
    ([$name:expr, $tensors:expr]) => {
        1
    };
    () => {
        0
    };
}

/// load tensors from a file
/// this macro uses multiple functions internally, slicing uses the same syntax as numpy
///
/// `[:]` and `[::]`: load all the elements along the corresponding dimension
///
/// `[1:]`: load from the first element to the end along the corresponding dimension
///
/// `[:10]`: load from the beginning to index 9 along the corresponding dimension
///
/// `[1:10]`: load from index 1 to index 9 along the corresponding dimension
///
/// `[1:10:2]`: load from index 1 to index 9 with step 2 along the corresponding dimension
///
/// `[1:10:2, 2:10:3]`: load from index 1 to index 9 with step 2 for the first dimension, and load from index 2 to index 9 with step 3 for the second dimension
///
/// `[::2]`: load all the elements with step 2 along the corresponding dimension
///
/// ## Examples:
/// ```
/// use tensor_core::prelude::*;
/// load!("test4.ft", f32, ["a", [:]])?; // load all the data of tensor "a" from the uncompress file
/// load!("test4.ft", f32, ["a", [:2]], ["data", [:2:3]])?; // load the first 32 elements from the uncompress file
/// ```
#[macro_export]
macro_rules! load {
    (
        $file_name:expr,
        $generic:ident,
        [
            $name:expr,
            [$($indexes:tt)*]
        ]
    ) => {
        {
        use std::io::Read;
        let mut file = std::fs::File::open($file_name)?;
        let mut buffer = vec![0u8; 1];
        file.read_exact(&mut buffer)?;
        let id = std::str::from_utf8(&buffer).unwrap();
        match id {
            "0" => {
            $crate::load_compressed_slice_gz_single::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }>
            ($file_name, ($name, &match_selection!($($indexes)*)))
            }
            "1" => {
                $crate::load_compressed_slice_deflate_single::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }>
            ($file_name, ($name, &match_selection!($($indexes)*)))
            }
            "2" => {
            $crate::load_compressed_slice_zlib_single::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }>
            ($file_name, ($name, &match_selection!($($indexes)*)))
            }
            "F" => {
            $crate::load_slice_single::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }>
            ($file_name, ($name, &match_selection!($($indexes)*)))
            }
            _ => {
                anyhow::bail!("invalid file format")
            }
        }
    }
    };

    (
        $file_name:expr,
        $generic:ident,
        $([
            $name:expr,
            [$($indexes:tt)*]
        ]),*
    ) => {
        {
        use std::io::Read;
        let mut file = std::fs::File::open($file_name)?;
        let mut buffer = vec![0u8; 1];
        file.read_exact(&mut buffer)?;
        let id = std::str::from_utf8(&buffer).unwrap();
        match id {
            "0" => {
            $crate::load_compressed_slice_gz::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }, {$crate::count_args!($($name =>[$($indexes)*]),*)}>
            ($file_name, [$(($name, &$crate::match_selection!($($indexes)*)), )*])
            }
            "1" => {
            $crate::load_compressed_slice_deflate::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }, {$crate::count_args!($($name =>[$($indexes)*]),*)}>
            ($file_name, [$(($name, &$crate::match_selection!($($indexes)*)), )*])
            }
            "2" => {
            $crate::load_compressed_slice_zlib::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }, {$crate::count_args!($($name =>[$($indexes)*]),*)}>
            ($file_name, [$(($name, &$crate::match_selection!($($indexes)*)), )*])
            }
            "F" => {
            $crate::load_slice::<$generic, Tensor<$generic>, { std::mem::size_of::<$generic>() }, {$crate::count_args!($($name =>[$($indexes)*]),*)}>
            ($file_name, [$(($name, &$crate::match_selection!($($indexes)*)), )*])
            }
            _ => {
                anyhow::bail!("invalid file format")
            }
        }
    }
    };
}

/// a macro for saving tensors to a file
/// ## Save Raw Tensors:
/// ```
/// use tensor_core::prelude::*;
/// let a: Tensor<f32> = Tensor::rand([128, 128, 128]).unwrap();
/// let b: Tensor<f32> = Tensor::rand([128, 128, 128]).unwrap();
/// save!("test4.ft", ["a", a.clone()]); // save raw tensors with key "a"
/// save!("test4.ft", ["a", a], ["b", b]); // save raw tensors with key "a"
/// ````
///
/// ## Save Tensors With Compression(Default Compression Level = 4):
/// ```
/// let a: Tensor<f32> = Tensor::rand([128, 128, 128]).unwrap();
/// let b: Tensor<f32> = Tensor::rand([128, 128, 128]).unwrap();
///
/// save!("test4.ftz", f32, ["data", a.clone()], compression = gzip); // save with key "data"
/// save!("test4.ftz", f32, ["data", a.clone()], compression = deflate);
/// save!("test4.ftz", f32, ["data", a.clone()], compression = zlib);
///
/// save!("test4.ftz", f32, ["data", a.clone()], ["data2", b.clone()], compression = gzip); // save with key "data", "data2"
/// save!("test4.ftz", f32, ["data", a.clone()], ["data2", b.clone()], compression = deflate);
/// save!("test4.ftz", f32, ["data", a], ["data2", b], compression = zlib);
/// ```
///
/// ## Save Tensors With Compression and Compression Level:
/// ```
/// let a: Tensor<f32> = Tensor::rand([128, 128, 128]).unwrap();
/// let b: Tensor<f32> = Tensor::rand([128, 128, 128]).unwrap();
/// // save with key "data"
/// save!("test4.ftz", f32, ["data", a.clone()], compression = gzip, compression_lvl = 9);
/// save!("test4.ftz", f32, ["data", a.clone()], compression = deflate, compression_lvl = 9);
/// save!("test4.ftz", f32, ["data", a.clone()], compression = zlib, compression_lvl = 9);
/// // save with key "data", "data2"
/// save!("test4.ftz", f32, ["data", a.clone()], ["data2", b.clone()], compression = gzip, compression_lvl = 9);
/// save!("test4.ftz", f32, ["data", a.clone()], ["data2", b.clone()], compression = deflate, compression_lvl = 9);
/// save!("test4.ftz", f32, ["data", a], ["data2", b], compression = zlib, compression_lvl = 9);
/// ```
#[macro_export]
macro_rules! save {
    ($file_name:expr, $([$name:expr, $tensors:expr]),*) => {
        $crate::save($file_name, [$(($name, $tensors)), *])
    };

    ($file_name:expr, $tensors:ident) => {
        compile_error!("please use [name, tensor] to save a tensor")
    };

    // macros for gzencoder
    ($file_name:expr, $generic:ident, $([$name:expr, $tensors:expr]),*, compression = gzip) => {
        save_compressed::<4, _, _, flate2::write::GzEncoder<Vec<u8>>,
         {std::mem::size_of::<$generic>()}, {count_args!($([$name, $tensors]),*)}>
        ($file_name, [$(($name, $tensors), ) *])
    };
    (
        $file_name:expr,
        $generic:ident,
        $([$name:expr, $tensors:expr]),*,
        compression = gzip,
        compression_lvl = $val:expr
    ) => {
        save_compressed::<$val, _, _, flate2::write::GzEncoder<Vec<u8>>,
         {std::mem::size_of::<$generic>()}, {count_args!($([$name, $tensors]),*)}>
        ($file_name, [$(($name, $tensors), ) *])
    };

    // macros for deflateencoder
    ($file_name:expr, $generic:ident, $([$name:expr, $tensors:expr]),*, compression = deflate) => {
        save_compressed::<4, _, _, flate2::write::DeflateEncoder<Vec<u8>>,
         {std::mem::size_of::<$generic>()}, {count_args!($([$name, $tensors]),*)}>
        ($file_name, [$(($name, $tensors), ) *])
    };
    (
        $file_name:expr,
        $generic:ident,
        $([$name:expr, $tensors:expr]),*,
        compression = deflate,
        compression_lvl = $val:expr
    ) => {
        save_compressed::<$val, _, _, flate2::write::DeflateEncoder<Vec<u8>>,
         {std::mem::size_of::<$generic>()}, {count_args!($([$name, $tensors]),*)}>
        ($file_name, [$(($name, $tensors), ) *])
    };

    // macros for zlib
    ($file_name:expr, $generic:ident, $([$name:expr, $tensors:expr]),*, compression = zlib) => {
        save_compressed::<4, _, _, flate2::write::ZlibEncoder<Vec<u8>>,
         {std::mem::size_of::<$generic>()}, {count_args!($([$name, $tensors]),*)}>
        ($file_name, [$(($name, $tensors), ) *])
    };
    (
        $file_name:expr,
        $generic:ident,
        $([$name:expr, $tensors:expr]),*,
        compression = zlib,
        compression_lvl = $val:expr
    ) => {
        save_compressed::<$val, _, _, flate2::write::ZlibEncoder<Vec<u8>>,
         {std::mem::size_of::<$generic>()}, {count_args!($([$name, $tensors]),*)}>
        ($file_name, [$(($name, $tensors), ) *])
    };
}

fn generate_header<A: CommonBounds, T: TensorInfo<A>>(tensors: &[(String, T)]) -> Result<String> {
    let mut begin = "FASTTENSORFASTTENSORHPLACEHOLD".len();
    let infos: Vec<String> = tensors
        .iter()
        .map(|(name, tensor)| {
            let string = format!(
                r#""{}":{{"begin":{},"name":"{}","shape":{:?},"strides":{:?},"size":{},"contiguous":{},"indices":[]}}"#,
                name,
                begin,
                name,
                tensor.shape(),
                tensor.strides(),
                tensor.size(),
                tensor.is_contiguous()
            );
            begin += tensor.size() * std::mem::size_of::<T>();
            string
        })
        .collect();
    return Ok(format!(r#"{{{}}}"#, infos.join(",")));
}

fn generate_header_compressed<T: CommonBounds, B: TensorInfo<T>>(
    tensors: &[(String, B)]
) -> Result<(Vec<String>, Vec<(String, usize, usize, usize)>)> {
    let infos: Vec<String> = tensors
        .iter()
        .map(|(name, tensor)| {
            format!(
                r#""{}":{{"begin":FASTTENSOR_PLACEHOLD,"name":"{}","shape":{:?},"strides":{:?},"size":{},"contiguous":{},"indices":place_holder0}}"#,
                name,
                name,
                tensor.shape(),
                tensor.shape().to_strides(),
                tensor.size(),
                tensor.is_contiguous()
            )
        })
        .collect();
    let res: Vec<(String, usize, usize, usize)> = tensors
        .iter()
        .map(|(name, x)| {
            let outer = x.size() / (*x.shape().last().unwrap() as usize);
            let inner = (*x.shape().last().unwrap() as usize) * std::mem::size_of::<T>();
            let num_chunks;
            let num_lines;
            let mut remain = 0;
            if x.size() * std::mem::size_of::<T>() < CHUNK_BUFF {
                num_chunks = 1;
                num_lines = outer;
            } else {
                num_lines = CHUNK_BUFF / inner;
                remain = outer % num_lines;
                num_chunks = outer / num_lines;
            }
            (name.clone(), num_chunks, num_lines, remain)
        })
        .collect::<Vec<(String, usize, usize, usize)>>();

    return Ok((infos, res));
}

pub fn save<
    'a,
    A: CommonBounds + ToBytes<Bytes = [u8; N]> + std::fmt::Display,
    const N: usize,
    const M: usize,
    B: TensorInfo<A> + Clone + std::marker::Send + 'static
>(file_name: &'a str, tensors: [(&'a str, B); M]) {
    let file_name = file_name.to_string();
    let new_tensors: Vec<(String, B)> = tensors
        .iter()
        .map(|(name, tensor)| (name.to_string(), tensor.clone()))
        .collect::<Vec<_>>();
    let file;
    if file_name.contains(".ftz") || !file_name.contains(".ft") {
        file = std::fs::File::create(format!("{}.ft", file_name)).unwrap();
    } else {
        file = std::fs::File::create(file_name).unwrap();
    }
    let mut writer = BufWriter::with_capacity(CHUNK_BUFF, file);
    writer.write_all("FASTTENSORFASTTENSORHPLACEHOLD".as_bytes()).unwrap();
    for (_, tensor) in &new_tensors {
        let last_stride: i64 = *tensor.strides().last().unwrap() as i64;
        let mut prg = vec![0; tensor.shape().len() - 1];
        let mut shape = tensor.shape().to_vec();
        shape.iter_mut().for_each(|x| {
            *x -= 1;
        });
        let inner_loop_size = *tensor.shape().last().unwrap() as i64;
        let outer_loop_size = tensor.size() / (inner_loop_size as usize);
        let mut ptr = tensor.ptr();
        for _ in 0..outer_loop_size {
            for i in 0..inner_loop_size {
                let val = ptr[i * last_stride];
                writer.write_all(&val.to_ne_bytes()).unwrap();
            }
            for j in (0..shape.len() - 1).rev() {
                if prg[j] < shape[j] {
                    prg[j] += 1;
                    ptr.offset(tensor.strides()[j]);
                    break;
                } else {
                    prg[j] = 0;
                    ptr.offset(-tensor.strides()[j] * shape[j]);
                }
            }
        }
    }
    writer.flush().unwrap();
    let current_pos = writer.seek(std::io::SeekFrom::Current(0)).unwrap();
    writer.seek(std::io::SeekFrom::Start(0)).unwrap();
    writer.write_all(format!("FASTTENSOR{:20}", current_pos).as_bytes()).unwrap();
    writer.seek(std::io::SeekFrom::Start(current_pos)).unwrap();
    let header = generate_header(&new_tensors).unwrap();
    writer.write_all(header.as_bytes()).unwrap();
    writer.flush().unwrap();
}

/// method to compress the tensor and save to file
///
/// `file_name`: name of the file to create
///
/// `tensors`: a list of tuples, [(name, tensor), ...]. Name will be used as the key to load the tensor
pub fn save_compressed<
    'a,
    const L: u32,
    T: CommonBounds + ToBytes<Bytes = [u8; N]>,
    B: TensorInfo<T>,
    W: CompressionTrait<L>,
    const N: usize,
    const M: usize
>(file_name: &'a str, tensors: [(&'a str, B); M]) {
    let new_tensors: Vec<(String, B)> = tensors
        .into_iter()
        .map(|(name, tensor)| (name.to_string(), tensor))
        .collect::<Vec<_>>();

    // initialize progress bar based on total element size of the data to save

    ////////////////////////////////////////////initialize progress bar////////////////////////////////////////////
    let total_size: usize = new_tensors
        .iter()
        .map(|(_, tensor)| tensor.size())
        .sum::<usize>();

    let pb: ProgressBar = ProgressBar::new(total_size as u64);
    pb.set_style(
        indicatif::ProgressStyle
            ::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
    );
    ////////////////////////////////////////////initialize progress bar////////////////////////////////////////////

    // generate compresse file header
    // (Vec<String>, list of tensor attributes: {"begin": 0, "name": "a", "shape": [1, 2, 3], ...}, ...),
    // will be used to deserialize and get the storing information when loading the file
    // Vec<(String, usize, usize, usize)>:  (tensor name, num_chunks, num_lines of each chunk, remain lines(if it is not divisable by CHUNK_BUFF)))
    let mut v: (Vec<String>, Vec<(String, usize, usize, usize)>) = generate_header_compressed(
        &new_tensors
    ).unwrap();
    let mut file;
    if !file_name.contains(".ftz") {
        file = std::fs::File::create(format!("{}.ftz", file_name)).unwrap();
    } else {
        file = std::fs::File::create(file_name).unwrap();
    }

    // write FASTTENSOR Magic Header, FASTTENSORHPLACEHOLD is a place holder for the location to the real header
    // real header is at the end of the file.
    // For Example: FASTTENSOR210123456789{"a": {"begin": 0, "name": "a", "shape": [1, 2, 3], ...}, ...}
    // 21 is the location of the real header, 0123456789 is the data we store.
    file.write_all(format!("{}FASTTENSORFASTTENSORHPLACEHOLD", W::id()).as_bytes()).unwrap();
    let mut attrs: Vec<Vec<(usize, usize, usize, usize)>> = vec![];
    for (idx, (_, tensor)) in new_tensors.iter().enumerate() {
        let last_stride: i64 = *tensor.strides().last().unwrap() as i64;
        let mut prg: Vec<i64> = vec![0; tensor.shape().len() - 1];
        let mut shape: Vec<i64> = tensor
            .shape()
            .iter()
            .map(|x| *x as i64)
            .collect();
        shape.iter_mut().for_each(|x: &mut i64| {
            *x -= 1;
        });
        let inner_loop_size: usize = *tensor.shape().last().unwrap() as usize;
        let mut ptr: Pointer<T> = tensor.ptr();
        let line_num: usize = v.1[idx].2;
        let num_chunks: usize = v.1[idx].1;
        let mut attributes: Vec<(usize, usize, usize, usize)> = vec![];
        let mut chunk: Vec<u8> = vec![0u8; CHUNK_BUFF];
        let mut len_so_far: usize = "1FASTTENSORFASTTENSORHPLACEHOLD".len();

        for k in 0..num_chunks {
            for j in 0..line_num {
                for i in 0..inner_loop_size {
                    let val = ptr[(i as i64) * last_stride];
                    let start = (j * inner_loop_size + i) * std::mem::size_of::<T>();
                    let end = (j * inner_loop_size + i + 1) * std::mem::size_of::<T>();
                    chunk[start..end].copy_from_slice(&val.to_ne_bytes());
                }
                pb.inc(inner_loop_size as u64);
                for h in (0..shape.len() - 1).rev() {
                    if prg[h] < shape[h] {
                        prg[h] += 1;
                        ptr.offset(tensor.strides()[h]);
                        break;
                    } else {
                        prg[h] = 0;
                        ptr.offset(-tensor.strides()[h] * (shape[h] as i64));
                    }
                }
            }
            let mut encoder = W::new();
            encoder.write_all_data(&chunk).unwrap();
            encoder.flush_all().unwrap();
            file.write_all(encoder.get_inner()).unwrap();
            attributes.push((
                k * line_num,
                len_so_far,
                encoder.get_inner().len(),
                line_num * inner_loop_size * std::mem::size_of::<T>(),
            ));
            len_so_far += encoder.get_inner().len();
        }
        let remain_outer: usize = v.1[idx].3;
        let mut remain_chunk: Vec<u8> =
            vec![0u8; remain_outer * inner_loop_size * std::mem::size_of::<T>()];
        let mut encoder = W::new();
        for j in 0..remain_outer {
            for i in 0..inner_loop_size {
                let val = ptr[(i as i64) * last_stride];
                let start = (j * inner_loop_size + i) * std::mem::size_of::<T>();
                let end = (j * inner_loop_size + i + 1) * std::mem::size_of::<T>();
                remain_chunk[start..end].copy_from_slice(&val.to_ne_bytes());
            }
            pb.inc(inner_loop_size as u64);
            for h in (0..shape.len() - 1).rev() {
                if prg[h] < shape[h] {
                    prg[h] += 1;
                    ptr.offset(tensor.strides()[h]);
                    break;
                } else {
                    prg[h] = 0;
                    ptr.offset(-tensor.strides()[h] * (shape[h] as i64));
                }
            }
        }
        encoder.write_all_data(&remain_chunk).unwrap();
        encoder.flush_all().unwrap();
        file.write_all(encoder.get_inner()).unwrap();
        attributes.push((
            num_chunks * line_num,
            len_so_far,
            encoder.get_inner().len(),
            v.1[idx].3 * inner_loop_size * std::mem::size_of::<T>(),
        ));
        v.0[idx] = v.0[idx].replace("place_holder0", &format!("{:?}", attributes));
        attrs.push(attributes);
    }

    // record the location of the end of the file so that we can come back and write the real header
    let current_pos = file.seek(std::io::SeekFrom::Current(0)).unwrap();

    // we go to the beginning of the file and correct the header
    file.seek(std::io::SeekFrom::Start(0)).unwrap();
    file.write_all(format!("{}FASTTENSOR{:20}", W::id(), current_pos).as_bytes()).unwrap();

    // jump back and start writing the real header
    file.seek(std::io::SeekFrom::Start(current_pos)).unwrap();
    let mut header_len = "1FASTTENSORFASTTENSORHPLACEHOLD".len();
    let length: Vec<usize> = attrs
        .iter()
        .map(|x| {
            let tmp = header_len;
            header_len += x.last().unwrap().1 + x.last().unwrap().2;
            tmp
        })
        .collect();

    v.0
        .iter_mut()
        .zip(length)
        .for_each(|(x, y)| {
            *x = x.replace("FASTTENSOR_PLACEHOLD", &format!("{:20}", y));
        });

    let to_write: String = format!(r#"{{{}}}"#, v.0.join(","));
    let to_write: String = to_write.replace("(", "[");
    let to_write: String = to_write.replace(")", "]");
    let bytes: &[u8] = to_write.as_bytes();
    file.write_all(bytes).unwrap();
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HeaderInfos {
    pub begin: u64,
    pub infos: Vec<HeaderInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HeaderInfo {
    pub begin: u64,
    pub name: String,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub size: usize,
    pub contiguous: bool,
    pub indices: Vec<(usize, usize, usize, usize)>,
}

/// load tensors from a uncompression file
/// `file_name`: name of the file to load
/// `queries`: a list of tensor names to load
/// # Examples
/// ```
/// use tensor_core::prelude::*;
/// let a: Tensor<f32> = Tensor::rand([128, 128, 128]).unwrap();
/// save!("test4.ft", ["a", a]);
/// load!("test4.ft", f32, a);
/// ```
pub fn __load<
    'a,
    T: CommonBounds + FromBytes<Bytes = [u8; N]>,
    B: TensorInfo<T> + TensorCreator<T> + Clone,
    const N: usize,
    const M: usize
>(file_name: &str, queries: [&str; M]) -> Result<Vec<B>> {
    let (begin, mut infos) = HeaderInfo::parse_header(&file_name)?;
    let mut tensors: Vec<B> = Vec::with_capacity(infos.len());
    queries.into_iter().try_for_each(|name| {
        let info = infos.get_mut(&name.to_string()).unwrap();
        let tensor: B = B::empty(&info.shape)?;
        tensors.push(tensor.clone());
        let mut file = std::fs::File::open(&file_name)?;
        file.seek(std::io::SeekFrom::Start(info.begin))?;
        let buffer = unsafe {
            std::slice::from_raw_parts_mut(
                tensor.ptr().ptr as *mut u8,
                tensor.size() * std::mem::size_of::<T>()
            )
        };
        file.read_exact(buffer)?;
        return Ok(());
    })?;
    let mut file = std::fs::File::open(&file_name)?;
    file.seek(std::io::SeekFrom::Start(begin as u64))?;
    Ok(tensors)
}

pub fn load_slice<
    'a,
    T: CommonBounds + FromBytes<Bytes = [u8; N]>,
    B: TensorCreator<T> + Clone + TensorInfo<T>,
    const N: usize,
    const M: usize
>(file: &str, queries: [(&str, &[Slice]); M]) -> Result<Vec<B>> {
    let (_, mut infos) = HeaderInfo::parse_header(&file)?;
    let mut tensors: Vec<B> = Vec::with_capacity(infos.len());
    queries.into_iter().try_for_each(|(name, slices)| {
        let info = infos.get_mut(&name.to_string()).unwrap();
        let mut file = std::fs::File::open(&file)?;
        file.seek(std::io::SeekFrom::Start(info.begin))?;

        let (mut res_shape, res_strides, offset) = slice_process(
            info.shape.clone(),
            info.strides.clone(),
            slices,
            std::mem::size_of::<T>() as i64
        )?;
        let mut shape = res_shape.clone();
        shape.iter_mut().for_each(|x| {
            *x /= std::mem::size_of::<T>() as i64;
        });
        let tensor: B = B::zeros(&shape)?;

        let size = shape.iter().product::<i64>() * (std::mem::size_of::<T>() as i64);
        let inner_loop_size = *res_shape.last().unwrap() as usize;
        let outer_loop_size = (size as usize) / (inner_loop_size as usize);
        tensors.push(tensor.clone());
        res_shape.iter_mut().for_each(|x| {
            if *x > 0 {
                *x -= 1;
            }
        });
        file.seek(std::io::SeekFrom::Current(offset as i64))?;
        let ptr = tensor.ptr().ptr as *mut u8;
        let buffer = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size as usize) };

        let mut index = 0;
        let mut prg = vec![0; res_shape.len() - 1];
        let last_stride: i64 = *res_strides.last().unwrap();
        if last_stride == 1 {
            for _ in 0..outer_loop_size {
                file.read_exact(&mut buffer[index..index + inner_loop_size])?;
                file.seek(std::io::SeekFrom::Current(-(inner_loop_size as i64)))?;
                index += inner_loop_size as usize;
                for j in (0..res_shape.len() - 1).rev() {
                    if prg[j] < res_shape[j] {
                        prg[j] += 1;
                        file.seek(std::io::SeekFrom::Current(res_strides[j] as i64))?;
                        break;
                    } else {
                        prg[j] = 0;
                        file.seek(
                            std::io::SeekFrom::Current(
                                -((res_strides[j] as i64) * (res_shape[j] as i64))
                            )
                        )?;
                    }
                }
            }
        } else {
            let inner_loop_size = inner_loop_size / std::mem::size_of::<T>();
            for _ in 0..outer_loop_size {
                for _ in 0..inner_loop_size {
                    file.read_exact(&mut buffer[index..index + std::mem::size_of::<T>()])?;
                    index += std::mem::size_of::<T>();
                    file.seek(
                        std::io::SeekFrom::Current(
                            (last_stride - (std::mem::size_of::<T>() as i64)) as i64
                        )
                    )?;
                }
                file.seek(
                    std::io::SeekFrom::Current(-((inner_loop_size as i64) * (last_stride as i64)))
                )?;
                for j in (0..res_shape.len() - 1).rev() {
                    if prg[j] < res_shape[j] {
                        prg[j] += 1;
                        file.seek(std::io::SeekFrom::Current(res_strides[j] as i64))?;
                        break;
                    } else {
                        prg[j] = 0;
                        file.seek(
                            std::io::SeekFrom::Current(
                                -((res_strides[j] as i64) * (res_shape[j] as i64))
                            )
                        )?;
                    }
                }
            }
        }
        return Ok(());
    })?;
    Ok(tensors)
}

pub fn load_slice_single<
    'a,
    T: CommonBounds + FromBytes<Bytes = [u8; N]>,
    B: TensorCreator<T> + Clone + TensorInfo<T>,
    const N: usize
>(file: &str, queries: (&str, &[Slice])) -> Result<B> {
    let (_, mut infos) = HeaderInfo::parse_header(&file)?;
    let (name, slices) = queries;
    let info = infos
        .get_mut(&name.to_string())
        .expect(
            &format!(
                r#"invalid query name when trying to load tensor "{}" in file "{}"."#,
                name,
                file
            )
        );
    let mut file = std::fs::File::open(&file)?;
    file.seek(std::io::SeekFrom::Start(info.begin))?;

    let (mut res_shape, res_strides, offset) = slice_process(
        info.shape.clone(),
        info.strides.clone(),
        slices,
        std::mem::size_of::<T>() as i64
    )?;
    let mut shape = res_shape.clone();
    shape.iter_mut().for_each(|x| {
        *x /= std::mem::size_of::<T>() as i64;
    });
    let tensor: B = B::empty(&shape)?;

    let size = shape.iter().product::<i64>() * (std::mem::size_of::<T>() as i64);
    let inner_loop_size = *res_shape.last().unwrap() as usize;
    let outer_loop_size = (size as usize) / (inner_loop_size as usize);
    res_shape.iter_mut().for_each(|x| {
        *x /= std::mem::size_of::<T>() as i64;
        if *x > 0 {
            *x -= 1;
        }
    });
    file.seek(std::io::SeekFrom::Current(offset as i64))?;
    let ptr = tensor.ptr().ptr as *mut u8;
    let buffer = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size as usize) };

    let mut index = 0;
    let mut prg = vec![0; res_shape.len() - 1];
    let last_stride: i64 = *res_strides.last().unwrap();
    if last_stride == (std::mem::size_of::<T>() as i64) {
        for _ in 0..outer_loop_size {
            file.read_exact(&mut buffer[index..index + inner_loop_size])?;
            file.seek(std::io::SeekFrom::Current(-(inner_loop_size as i64)))?;
            index += inner_loop_size as usize;
            for j in (0..res_shape.len() - 1).rev() {
                if prg[j] < res_shape[j] {
                    prg[j] += 1;
                    file.seek(std::io::SeekFrom::Current(res_strides[j] as i64))?;
                    break;
                } else {
                    prg[j] = 0;
                    file.seek(
                        std::io::SeekFrom::Current(
                            -((res_strides[j] as i64) * (res_shape[j] as i64))
                        )
                    )?;
                }
            }
        }
    } else {
        let inner_loop_size = inner_loop_size / std::mem::size_of::<T>();
        for _ in 0..outer_loop_size {
            for _ in 0..inner_loop_size {
                file.read_exact(&mut buffer[index..index + std::mem::size_of::<T>()])?;
                index += std::mem::size_of::<T>();
                file.seek(
                    std::io::SeekFrom::Current(
                        (last_stride - (std::mem::size_of::<T>() as i64)) as i64
                    )
                )?;
            }
            file.seek(
                std::io::SeekFrom::Current(-((inner_loop_size as i64) * (last_stride as i64)))
            )?;
            for j in (0..res_shape.len() - 1).rev() {
                if prg[j] < res_shape[j] {
                    prg[j] += 1;
                    file.seek(std::io::SeekFrom::Current(res_strides[j] as i64))?;
                    break;
                } else {
                    prg[j] = 0;
                    file.seek(
                        std::io::SeekFrom::Current(
                            -((res_strides[j] as i64) * (res_shape[j] as i64))
                        )
                    )?;
                }
            }
        }
    }
    Ok(tensor)
}

pub fn load_compressed<
    T: CommonBounds + FromBytes<Bytes = [u8; N]>,
    B: TensorCreator<T> + Clone + TensorInfo<T>,
    const N: usize,
    const M: usize
>(file: String, queries: [&str; M]) -> Result<Vec<B>> {
    let res: HashMap<String, HeaderInfo> = HeaderInfo::parse_header_compressed(&file)?;
    let mut ret = Vec::with_capacity(res.len());
    queries.into_iter().try_for_each(|name| {
        let info = res.get(&name.to_string()).unwrap();
        let tensor: B = B::empty(info.shape.clone())?;

        let buffer = unsafe {
            std::slice::from_raw_parts_mut(
                tensor.ptr().ptr as *mut u8,
                tensor.size() * std::mem::size_of::<T>()
            )
        };
        let mut file = std::fs::File::open(&file)?;
        let inner_loop_size = (*info.shape.last().unwrap() as usize) * std::mem::size_of::<T>();
        info.indices.iter().for_each(|(offset, compressed_len, _, real_len)| {
            let start = *offset * inner_loop_size;
            let res = &mut buffer[start..start + *real_len];
            let mut vec = vec![0u8; *compressed_len];
            file.read_exact(&mut vec).unwrap();
            let mut decoder = flate2::read::GzDecoder::new(&vec[..]);
            decoder.read_exact(res).unwrap();
        });
        ret.push(tensor);
        return Ok(());
    })?;
    Ok(ret)
}

macro_rules! register_load_compress_slice {
    ($name:ident, $method:expr) => {
        pub fn $name<
            'a,
            T: CommonBounds + FromBytes<Bytes = [u8; N]>,
            B: TensorCreator<T> + Clone + TensorInfo<T>,
            const N: usize,
            const M: usize,
        >(
            file_name: &str,
            queries: [(&str, &[Slice]); M],
        ) -> Result<Vec<B>> {
            let res: HashMap<String, HeaderInfo> = HeaderInfo::parse_header_compressed(file_name)?;
            let mut ret = Vec::with_capacity(res.len());
            let mut file = File::open(file_name)?;

            queries.iter().try_for_each(|(name, slices)| {
                let info = res.get(*name).unwrap();
                let (mut res_shape, res_strides, offset) = slice_process(
                    info.shape.clone(),
                    info.strides.clone(),
                    slices,
                    std::mem::size_of::<T>() as i64,
                )?;
                let mut shape = res_shape.clone();
                shape.iter_mut().for_each(|x| {
                    *x /= std::mem::size_of::<T>() as i64;
                });
                let tensor: B = B::empty(&shape)?;

                let size = tensor.size() * std::mem::size_of::<T>();
                if size == 0 {
                    ret.push(tensor);
                    return Ok(());
                }
                let inner_loop_size =
                    (*info.shape.last().unwrap() as usize) * std::mem::size_of::<T>();
                let real_inner_loop_size =
                    (*info.shape.last().unwrap() as usize) * std::mem::size_of::<T>();

                let skip =
                    (offset % (inner_loop_size as i64)) / (std::mem::size_of::<T>() as i64);
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(tensor.ptr().ptr as *mut u8, size) };
                let row_idx = offset
                    / (*info.shape.last().unwrap() as i64)
                    / (std::mem::size_of::<T>() as i64);
                let last_stride = *res_strides.last().unwrap() as usize;

                let chunk_size = info.indices[0].3;
                let num_lines = chunk_size / real_inner_loop_size;
                let mut i = (row_idx as usize) / num_lines;
                let (guess_idx, idx, compressed_len, guess_real_len) = info.indices[i as usize];

                ///////////////////////////////////////////////////////////////////////////////////////////////////
                // offset for guess_idx
                file.seek(std::io::SeekFrom::Start(idx as u64))?;

                let gap = (row_idx as usize) - guess_idx;
                let mut jump = gap * inner_loop_size;

                // load the raw compressed data
                let mut compressed_vec = vec![0u8; compressed_len];
                file.read_exact(&mut compressed_vec)?;
                let mut decoder = $method(&compressed_vec[..]);
                decoder.read_exact(&mut vec![0u8; jump])?;

                // get the uncompressed data
                let mut data = &mut vec![0u8; guess_real_len - jump][..];
                decoder.read_exact(data)?;

                let mut prg = vec![0; res_shape.len() - 1];

                res_shape.iter_mut().for_each(|x| {
                    *x /= std::mem::size_of::<T>() as i64;
                });
                let offset_inner_loop_size = *res_shape.last().unwrap() as usize;
                let offset_outer_loop_size = tensor.size() / offset_inner_loop_size;
                res_shape.iter_mut().for_each(|x| {
                    *x -= 1;
                });
                data = &mut data[(skip as usize) * std::mem::size_of::<T>()..];
                let mut index = jump + (skip as usize) * std::mem::size_of::<T>();
                let mut holder;
                let mut buffer_traker = 0;
                let mut next_real_len = guess_real_len;
                let mut index_offset = index;
                for _ in 0..offset_outer_loop_size {
                    let index_copy = index;
                    for _ in 0..offset_inner_loop_size {
                        //get value
                        let val = &data
                            [index - index_offset..index - index_offset + std::mem::size_of::<T>()];
                        index += last_stride;
                        buffer[buffer_traker..buffer_traker + std::mem::size_of::<T>()]
                            .copy_from_slice(val);
                        let _v = T::from_ne_bytes(val.try_into().unwrap());
                        buffer_traker += std::mem::size_of::<T>();
                    }
                    index = index_copy;
                    for j in (0..res_shape.len() - 1).rev() {
                        if prg[j] < res_shape[j] {
                            prg[j] += 1;
                            index += res_strides[j] as usize;
                            break;
                        } else {
                            prg[j] = 0;
                            index -= (res_strides[j] as usize) * (res_shape[j] as usize);
                        }
                    }
                    if index >= next_real_len {
                        let (
                            mut _next_guess_idx,
                            mut next_idx,
                            mut next_compressed_len,
                            mut next_guess_real_len,
                        ) = info.indices[(i as usize) + 1];
                        jump = index - next_real_len;
                        if jump > next_guess_real_len {
                            jump = index;
                            let remainder = jump % next_guess_real_len;
                            jump -= remainder;
                            let chunks = jump / next_guess_real_len;
                            i = chunks;
                            jump = remainder;
                            (
                                _next_guess_idx,
                                next_idx,
                                next_compressed_len,
                                next_guess_real_len,
                            ) = info.indices[i];
                            next_real_len = chunks * next_guess_real_len;
                            next_real_len += next_guess_real_len;
                        } else {
                            i += 1;
                            next_real_len += next_guess_real_len;
                        }
                        let mut next_compressed_vec = vec![0u8; next_compressed_len];
                        file.seek(std::io::SeekFrom::Start(next_idx as u64))?; // offset for next guess_idx
                        file.read_exact(&mut next_compressed_vec)?;

                        let mut next_decoder = $method(&next_compressed_vec[..]);
                        let mut next_data;
                        if jump > next_guess_real_len {
                            next_decoder.read_exact(&mut vec![0u8; next_guess_real_len])?;
                            next_data = vec![0u8; next_guess_real_len - jump];
                        } else {
                            next_decoder.read_exact(&mut vec![0u8; jump])?;
                            next_data = vec![0u8; next_guess_real_len - jump];
                        }
                        next_decoder.read_exact(&mut next_data)?;
                        holder = next_data;
                        data = &mut holder[(skip as usize) * std::mem::size_of::<T>()..];
                        index_offset = index;
                    }
                }
                ///////////////////////////////////////////////////////////////////////////////////////////////////
                ret.push(tensor);
                return Ok(());
            })?;
            Ok(ret)
        }
    };

    (single, $name:ident, $method:expr) => {
        pub fn $name<
            'a,
            T: CommonBounds + FromBytes<Bytes = [u8; N]>,
            B: TensorCreator<T> + Clone + TensorInfo<T>,
            const N: usize,
        >(
            file_name: &str,
            queries: (&str, &[Slice]),
        ) -> Result<B> {
            let res: HashMap<String, HeaderInfo> = HeaderInfo::parse_header_compressed(file_name)?;
            let mut file = File::open(file_name)?;

            let (name, slices) = queries;
            let info = res.get(name).unwrap();
            let (mut res_shape, res_strides, offset) = slice_process(
                info.shape.clone(),
                info.strides.clone(),
                slices,
                std::mem::size_of::<T>() as i64,
            )?;
            let mut shape = res_shape.clone();
            shape.iter_mut().for_each(|x| {
                *x /= std::mem::size_of::<T>() as i64;
            });
            let tensor: B = B::empty(&shape)?;

            let size = tensor.size() * std::mem::size_of::<T>();
            if size == 0 {
                return Ok(tensor);
            }
            let inner_loop_size = (*info.shape.last().unwrap() as usize) * std::mem::size_of::<T>();
            let real_inner_loop_size =
                (*info.shape.last().unwrap() as usize) * std::mem::size_of::<T>();

            let skip = (offset % (inner_loop_size as i64)) / (std::mem::size_of::<T>() as i64);
            let buffer =
                unsafe { std::slice::from_raw_parts_mut(tensor.ptr().ptr as *mut u8, size) };
            let row_idx = offset
                / (*info.shape.last().unwrap() as i64)
                / (std::mem::size_of::<T>() as i64);
            let last_stride = *res_strides.last().unwrap() as usize;

            let chunk_size = info.indices[0].3;
            let num_lines = chunk_size / real_inner_loop_size;
            let mut i = (row_idx as usize) / num_lines;
            let (guess_idx, idx, compressed_len, guess_real_len) = info.indices[i as usize];

            ///////////////////////////////////////////////////////////////////////////////////////////////////
            // offset for guess_idx
            file.seek(std::io::SeekFrom::Start(idx as u64))?;

            let gap = (row_idx as usize) - guess_idx;
            let mut jump = gap * inner_loop_size;

            // load the raw compressed data
            let mut compressed_vec = vec![0u8; compressed_len];
            file.read_exact(&mut compressed_vec)?;
            let mut decoder = $method(&compressed_vec[..]);
            decoder.read_exact(&mut vec![0u8; jump])?;

            // get the uncompressed data
            let mut data = &mut vec![0u8; guess_real_len - jump][..];
            decoder.read_exact(data)?;

            let mut prg = vec![0; res_shape.len() - 1];

            res_shape.iter_mut().for_each(|x| {
                *x /= std::mem::size_of::<T>() as i64;
            });
            let offset_inner_loop_size = *res_shape.last().unwrap() as usize;
            let offset_outer_loop_size = tensor.size() / offset_inner_loop_size;
            res_shape.iter_mut().for_each(|x| {
                *x -= 1;
            });
            data = &mut data[(skip as usize) * std::mem::size_of::<T>()..];
            let mut index = jump + (skip as usize) * std::mem::size_of::<T>();
            let mut holder;
            let mut buffer_traker = 0;
            let mut next_real_len = guess_real_len;
            let mut index_offset = index;
            for _ in 0..offset_outer_loop_size {
                let index_copy = index;
                for _ in 0..offset_inner_loop_size {
                    //get value
                    let val = &data
                        [index - index_offset..index - index_offset + std::mem::size_of::<T>()];
                    index += last_stride;
                    buffer[buffer_traker..buffer_traker + std::mem::size_of::<T>()]
                        .copy_from_slice(val);
                    let _v = T::from_ne_bytes(val.try_into().unwrap());
                    buffer_traker += std::mem::size_of::<T>();
                }
                index = index_copy;
                for j in (0..res_shape.len() - 1).rev() {
                    if prg[j] < res_shape[j] {
                        prg[j] += 1;
                        index += res_strides[j] as usize;
                        break;
                    } else {
                        prg[j] = 0;
                        index -= (res_strides[j] as usize) * (res_shape[j] as usize);
                    }
                }
                if index >= next_real_len {
                    let (
                        mut _next_guess_idx,
                        mut next_idx,
                        mut next_compressed_len,
                        mut next_guess_real_len,
                    ) = info.indices[(i as usize) + 1];
                    jump = index - next_real_len;
                    if jump > next_guess_real_len {
                        jump = index;
                        let remainder = jump % next_guess_real_len;
                        jump -= remainder;
                        let chunks = jump / next_guess_real_len;
                        i = chunks;
                        jump = remainder;
                        (
                            _next_guess_idx,
                            next_idx,
                            next_compressed_len,
                            next_guess_real_len,
                        ) = info.indices[i];
                        next_real_len = chunks * next_guess_real_len;
                        next_real_len += next_guess_real_len;
                    } else {
                        i += 1;
                        next_real_len += next_guess_real_len;
                    }
                    let mut next_compressed_vec = vec![0u8; next_compressed_len];
                    file.seek(std::io::SeekFrom::Start(next_idx as u64))?; // offset for next guess_idx
                    file.read_exact(&mut next_compressed_vec)?;

                    let mut next_decoder = $method(&next_compressed_vec[..]);
                    let mut next_data;
                    if jump > next_guess_real_len {
                        next_decoder.read_exact(&mut vec![0u8; next_guess_real_len])?;
                        next_data = vec![0u8; next_guess_real_len - jump];
                    } else {
                        next_decoder.read_exact(&mut vec![0u8; jump])?;
                        next_data = vec![0u8; next_guess_real_len - jump];
                    }
                    next_decoder.read_exact(&mut next_data)?;
                    holder = next_data;
                    data = &mut holder[(skip as usize) * std::mem::size_of::<T>()..];
                    index_offset = index;
                }
            }
            Ok(tensor)
        }
    };
}

register_load_compress_slice!(load_compressed_slice_gz, flate2::read::GzDecoder::new);
register_load_compress_slice!(load_compressed_slice_deflate, flate2::read::DeflateDecoder::new);
register_load_compress_slice!(load_compressed_slice_zlib, flate2::read::ZlibDecoder::new);
register_load_compress_slice!(
    single,
    load_compressed_slice_gz_single,
    flate2::read::GzDecoder::new
);
register_load_compress_slice!(
    single,
    load_compressed_slice_deflate_single,
    flate2::read::DeflateDecoder::new
);
register_load_compress_slice!(
    single,
    load_compressed_slice_zlib_single,
    flate2::read::ZlibDecoder::new
);

impl HeaderInfo {
    pub fn parse_header(file: &str) -> Result<(u64, HashMap<String, Self>)> {
        let file = File::open(file)?;
        let mut reader = std::io::BufReader::new(file);
        let mut buffer = [0; 10];
        reader.read_exact(&mut buffer)?;
        let header = std::str::from_utf8(&buffer)?;
        match header {
            "FASTTENSOR" => {}
            _ => {
                return Err(anyhow::anyhow!("not a fasttensor file"));
            }
        }
        let mut buffer2 = [0u8; 20];
        reader.read_exact(&mut buffer2)?;
        let length_str = std::str::from_utf8(&buffer2)?;
        let length: u64 = length_str.trim().parse::<u64>()?;
        reader.seek(std::io::SeekFrom::Start(length))?; // offset for header
        let mut buffer4: Vec<u8> = vec![];
        reader.read_to_end(&mut buffer4)?;
        let info = std::str::from_utf8(&buffer4)?;
        Ok((length, serde_json::from_str(info)?))
    }

    pub fn parse_header_compressed(file: &str) -> Result<HashMap<String, HeaderInfo>> {
        let mut file = File::open(file)?;
        file.read_exact(&mut [0u8; "1FASTTENSOR".len()])?;
        let mut header_infos = [0u8; 20];
        file.read_exact(&mut header_infos)?;
        let header = std::str::from_utf8(&header_infos)?;
        let header_int = header.trim().parse::<u64>()?;
        file.seek(std::io::SeekFrom::Start(header_int))?; // offset for header
        let mut buffer3 = vec![];
        file.read_to_end(&mut buffer3)?;
        let info = std::str::from_utf8(&buffer3)?;
        let ret: HashMap<String, HeaderInfo> = serde_json::from_str::<HashMap<String, Self>>(
            &info
        )?;
        Ok(ret)
    }
}
