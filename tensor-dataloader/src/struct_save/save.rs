use std::io::{Seek, Write};

use crate::compression_trait::{CompressionAlgo, DataLoaderTrait, Meta};
use crate::Endian;
use crate::{compression_trait::CompressionTrait, CHUNK_BUFF};
use flate2::write::{DeflateEncoder, GzEncoder};
use indicatif::ProgressBar;
use tensor_types::dtype::Dtype;

pub trait Save {
    type Meta: for<'a> serde::Deserialize<'a>;
    fn __save(
        data: &Self,
        file: &mut std::fs::File,
        len_so_far: &mut usize,
        global_cnt: &mut usize,
        compression_algo: CompressionAlgo,
        endian: Endian,
        level: u32,
    ) -> std::io::Result<Self::Meta>;
    fn save(&self, path: &str) -> std::io::Result<()>
    where
        <Self as Save>::Meta: serde::Serialize,
    {
        let mut file = std::fs::File::create(path)?;
        let meta = <Self as Save>::__save(
            &self,
            &mut file,
            &mut 0,
            &mut 0,
            CompressionAlgo::NoCompression,
            Endian::Native,
            9,
        )?;
        let serialized = serde_json::to_string(&meta)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }
}

fn generate_header_compressed(
    meta: &Meta,
) -> (
    (
        usize,                             /*begin */
        String,                            /* name */
        Vec<i64>,                          /* shape */
        Vec<i64>,                          /* strides */
        usize,                             /* size */
        Dtype,                             /* dtype */
        CompressionAlgo,                   /* compression_algo */
        Endian,                            /* endian */
        Vec<(usize, usize, usize, usize)>, /* indices */
    ),
    (String, usize, usize, usize, usize),
) {
    let info = (
        0usize,
        meta.name.clone(),
        meta.data_saver.shape().to_vec(),
        meta.data_saver.shape().to_strides().to_vec(),
        meta.data_saver.size(),
        meta.data_saver.dtype(),
        meta.compression_algo,
        meta.endian,
        vec![],
    );
    let res = {
        let x = &meta.data_saver;
        let outer = x.size() / (*x.shape().last().unwrap() as usize);
        let inner = (*x.shape().last().unwrap() as usize) * x.mem_size();
        let num_chunks;
        let mut num_lines;
        let mut remain = 0;
        let mut buffer_size;
        if x.size() * x.mem_size() < CHUNK_BUFF {
            num_chunks = 1;
            num_lines = outer;
            buffer_size = num_lines * inner;
        } else {
            buffer_size = ((CHUNK_BUFF - 1) / inner) * inner;
            num_lines = buffer_size / inner;
            if num_lines == 0 {
                num_lines = 1;
                buffer_size = inner;
            }
            remain = outer % num_lines;
            num_chunks = outer / num_lines;
        }
        (
            meta.name.clone(),
            num_chunks,
            num_lines,
            remain,
            buffer_size,
        )
    };

    (info, res)
}

/// method to compress the tensor and save to file
///
/// `file_name`: name of the file to create
///
/// `tensors`: a list of tuples, [(name, tensor), ...]. Name will be used as the key to load the tensor
pub fn save(
    file: &mut std::fs::File,
    mut meta: Meta,
    len: &mut usize,
    global_cnt: usize,
) -> std::io::Result<(
    usize,                             /*begin */
    String,                            /* name */
    Vec<i64>,                          /* shape */
    Vec<i64>,                          /* strides */
    usize,                             /* size */
    Dtype,                             /* dtype */
    CompressionAlgo,                   /* compression_algo */
    Endian,                            /* endian */
    Vec<(usize, usize, usize, usize)>, /* indices */
)> {
    // initialize progress bar based on total element size of the data to save
    ////////////////////////////////////////////initialize progress bar////////////////////////////////////////////
    let total_size: usize = meta.data_saver.size();

    let pb = ProgressBar::new(total_size as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap(),
    );
    ////////////////////////////////////////////initialize progress bar////////////////////////////////////////////

    // generate compresse file header
    // (Vec<String>, list of tensor attributes: {"begin": 0, "name": "a", "shape": [1, 2, 3], ...}, ...),
    // will be used to deserialize and get the storing information when loading the file
    // Vec<(String, usize, usize, usize)>:  (tensor name, num_chunks, num_lines of each chunk, remain lines(if it is not divisable by CHUNK_BUFF)))
    let (mut info, save_config) = generate_header_compressed(&meta);
    // write FASTTENSOR Magic Header, FASTTENSORHPLACEHOLD is a place holder for the location to the real header
    // real header is at the end of the file.
    // For Example: FASTTENSOR210123456789{"a": {"begin": 0, "name": "a", "shape": [1, 2, 3], ...}, ...}
    // 21 is the location of the real header, 0123456789 is the data we store.
    let mut len_so_far = "FASTTENSORFASTTENSORHPLACEHOLD".len();
    if global_cnt == 0 {
        file.write_all("FASTTENSORFASTTENSORHPLACEHOLD".as_bytes())?;
    } else {
        len_so_far = *len;
    }
    let last_stride: i64 = *meta.data_saver.strides().last().unwrap() as i64;
    let mut prg: Vec<i64> = vec![0; meta.data_saver.shape().len() - 1];
    let mut shape: Vec<i64> = meta.data_saver.shape().iter().map(|x| *x as i64).collect();
    shape.iter_mut().for_each(|x: &mut i64| {
        *x -= 1;
    });
    let inner_loop_size: usize = *meta.data_saver.shape().last().unwrap() as usize;
    let line_num: usize = save_config.2;
    let num_chunks: usize = save_config.1;
    let buffer_size: usize = save_config.4;
    let mut attributes = vec![];
    let mut chunk = vec![0u8; buffer_size];
    let unpack = get_unpack_closure(meta.endian);
    for k in 0..num_chunks {
        for j in 0..line_num {
            for i in 0..inner_loop_size {
                let start = (j * inner_loop_size + i) * meta.data_saver.mem_size();
                let end = (j * inner_loop_size + i + 1) * meta.data_saver.mem_size();
                unpack(
                    &mut meta.data_saver,
                    ((i as i64) * last_stride) as isize,
                    &mut chunk[start..end],
                );
            }
            pb.inc(inner_loop_size as u64);
            for h in (0..shape.len() - 1).rev() {
                if prg[h] < shape[h] {
                    prg[h] += 1;
                    meta.data_saver
                        .offset(meta.data_saver.strides()[h] as isize);
                    break;
                } else {
                    prg[h] = 0;
                    meta.data_saver
                        .offset(-(meta.data_saver.strides()[h] * (shape[h] as i64)) as isize);
                }
            }
        }
        compress_data(
            &meta,
            &chunk,
            file,
            &mut attributes,
            &mut len_so_far,
            k,
            line_num,
        )?;
    }
    let remain_outer: usize = save_config.3;
    let mut remain_chunk: Vec<u8> =
        vec![0u8; remain_outer * inner_loop_size * meta.data_saver.mem_size()];
    for j in 0..remain_outer {
        for i in 0..inner_loop_size {
            let start = (j * inner_loop_size + i) * meta.data_saver.mem_size();
            let end = (j * inner_loop_size + i + 1) * meta.data_saver.mem_size();
            unpack(
                &mut meta.data_saver,
                ((i as i64) * last_stride) as isize,
                &mut remain_chunk[start..end],
            );
        }
        pb.inc(inner_loop_size as u64);
        for h in (0..shape.len() - 1).rev() {
            if prg[h] < shape[h] {
                prg[h] += 1;
                meta.data_saver
                    .offset(meta.data_saver.strides()[h] as isize);
                break;
            } else {
                prg[h] = 0;
                meta.data_saver
                    .offset(-(meta.data_saver.strides()[h] * (shape[h] as i64)) as isize);
            }
        }
    }
    compress_data(
        &meta,
        &remain_chunk,
        file,
        &mut attributes,
        &mut len_so_far,
        num_chunks,
        line_num,
    )?;
    let current_pos = file.seek(std::io::SeekFrom::Current(0))?;
    file.seek(std::io::SeekFrom::Start(0))?;
    file.write_all(format!("FASTTENSOR{:20}", current_pos).as_bytes())?;
    file.seek(std::io::SeekFrom::Start(current_pos))?;
    let length = *len;
    *len += attributes.last().unwrap().1 + attributes.last().unwrap().2;
    info.8 = attributes;
    info.0 = length;

    Ok(info)
}

fn compress_data(
    meta: &Meta,
    chunk: &[u8],
    file: &mut std::fs::File,
    attributes: &mut Vec<(usize, usize, usize, usize)>,
    len_so_far: &mut usize,
    k: usize,
    line_num: usize,
) -> std::io::Result<()> {
    let mut closure = |compressed_data: &[u8]| -> std::io::Result<()> {
        file.write_all(compressed_data)?;
        attributes.push((
            k * line_num,
            *len_so_far,           /* start offset of the compressed data */
            compressed_data.len(), /* length of the compressed data */
            chunk.len(),           /* bytes of the data in the chunk */
        ));
        *len_so_far += compressed_data.len();
        Ok(())
    };

    match meta.compression_algo {
        CompressionAlgo::Gzip => {
            let mut encoder =
                GzEncoder::new(Vec::new(), flate2::Compression::new(meta.compression_level));
            encoder.write_all_data(chunk)?;
            encoder.flush_all()?;
            let compressed_data = encoder.finish_all()?;
            closure(&compressed_data)?
        }
        CompressionAlgo::Deflate => {
            let mut encoder =
                DeflateEncoder::new(Vec::new(), flate2::Compression::new(meta.compression_level));
            encoder.write_all_data(chunk)?;
            encoder.flush_all()?;
            let compressed_data = encoder.finish_all()?;
            closure(&compressed_data)?
        }
        CompressionAlgo::Zlib => {
            let mut encoder =
                DeflateEncoder::new(Vec::new(), flate2::Compression::new(meta.compression_level));
            encoder.write_all_data(chunk)?;
            encoder.flush_all()?;
            let compressed_data = encoder.finish_all()?;
            closure(&compressed_data)?
        }
        CompressionAlgo::NoCompression => closure(chunk)?,
    }
    Ok(())
}

fn get_unpack_closure(endian: Endian) -> impl Fn(&mut Box<dyn DataLoaderTrait>, isize, &mut [u8]) {
    match endian {
        Endian::Little => {
            |data_saver: &mut Box<dyn DataLoaderTrait>, offset: isize, data: &mut [u8]| {
                data_saver.fill_le_bytes_slice(offset, data)
            }
        }
        Endian::Big => {
            |data_saver: &mut Box<dyn DataLoaderTrait>, offset: isize, data: &mut [u8]| {
                data_saver.fill_be_bytes_slice(offset, data)
            }
        }
        Endian::Native => {
            |data_saver: &mut Box<dyn DataLoaderTrait>, offset: isize, data: &mut [u8]| {
                data_saver.fill_ne_bytes_slice(offset, data)
            }
        }
    }
}

macro_rules! impl_save {
    ($struct:ident) => {
        impl Save for $struct {
            type Meta = $struct;
            fn __save(
                data: &Self,
                _: &mut std::fs::File,
                _: &mut usize,
                _: &mut usize,
                _: CompressionAlgo,
                _: Endian,
                _: u32,
            ) -> std::io::Result<Self> {
                Ok(*data)
            }
        }
    };
}

impl_save!(bool);
impl_save!(i8);
impl_save!(i16);
impl_save!(i32);
impl_save!(i64);
impl_save!(u8);
impl_save!(u16);
impl_save!(u32);
impl_save!(u64);
impl_save!(f32);
impl_save!(f64);
